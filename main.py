import sys
import ssl
import certifi
import requests
import folium
import osmnx as ox
import networkx as nx
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from folium.plugins import HeatMap, MarkerCluster
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, \
    QPushButton, QComboBox, QCheckBox, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy, QListWidget, QListWidgetItem, \
    QDialog, QProgressBar, QGroupBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal
import os
import re
import logging
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Load crime data
crime_data = pd.read_csv('CleanedCrimeData.csv')

# Setup logging for debugging purposes
logging.basicConfig(filename='geocoding_debug.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Example crime weights dictionary
crime_weights = {
    'Anti-social behaviour': 1,
    'Bicycle theft': 2,
    'Burglary': 3,
    'Criminal damage and arson': 3,
    'Drugs': 2,
    'Other crime': 2,
    'Robbery': 4,
    'Vehicle crime': 3,
    'Violence and sexual offences': 5,
}


def calculate_crime_score(crime_data, crime_weights):
    """Calculate the crime score for each crime incident."""
    crime_data['Crime Score'] = crime_data['Crime type'].map(crime_weights)
    return crime_data


def generate_crime_grid(crime_data, grid_size=0.01):
    """Generate a grid over the map and calculate crime scores for each cell."""
    minx, miny, maxx, maxy = crime_data['Longitude'].min(), crime_data['Latitude'].min(), crime_data['Longitude'].max(), crime_data['Latitude'].max()
    x_range = np.arange(minx, maxx, grid_size)
    y_range = np.arange(miny, maxy, grid_size)
    grid_cells = [Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]) for x in x_range for y in y_range]
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'])

    # Assign crime scores to each grid cell
    grid['Crime Score'] = 0
    for idx, row in crime_data.iterrows():
        point = Point(row['Longitude'], row['Latitude'])
        matching_cell = grid[grid.contains(point)]
        if not matching_cell.empty:
            grid.loc[matching_cell.index, 'Crime Score'] += row['Crime Score']

    return grid


def modify_graph_with_crime_data(G, crime_grid, threshold):
    """Modify the graph's edge weights based on crime data."""
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_center = [(G.nodes[u]['y'] + G.nodes[v]['y']) / 2, (G.nodes[u]['x'] + G.nodes[v]['x']) / 2]
        point = Point(edge_center[1], edge_center[0])
        matching_cell = crime_grid[crime_grid.contains(point)]
        if not matching_cell.empty and matching_cell['Crime Score'].values[0] > threshold:
            data['length'] *= 10  # Penalize this route heavily

    return G


class LoadingDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculating Route")
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.setModal(True)

        layout = QVBoxLayout()
        self.label = QLabel("Calculating optimal route, please wait...")
        layout.addWidget(self.label)

        self.progress = QProgressBar(self)
        self.progress.setRange(0, 0)  # Indeterminate state
        layout.addWidget(self.progress)

        self.setLayout(layout)
        self.resize(300, 100)


class RouteCalculationThread(QThread):
    route_calculated = pyqtSignal(list, float, str)

    def __init__(self, start_coords, end_coords, stops_coords, mode, speed):
        super().__init__()
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.stops_coords = stops_coords
        self.mode = mode
        self.speed = speed

    def run(self):
        try:
            # Calculate the full route with stops
            all_coords = [self.start_coords] + self.stops_coords + [self.end_coords]
            route = self.get_full_route(all_coords, self.mode)
            total_distance_km = self.calculate_total_distance(route)
            travel_time_minutes = self.calculate_travel_time(total_distance_km, self.speed)

            self.route_calculated.emit(route, travel_time_minutes, self.mode)
        except Exception as e:
            self.route_calculated.emit([], 0, str(e))

    def get_full_route(self, points, mode):
        try:
            mode_map = {
                'walking': 'walk',
                'biking': 'bike',
                'driving': 'drive'
            }
            G = ox.graph_from_place('Southampton, England', network_type=mode_map[mode])

            route = []
            for i in range(len(points) - 1):
                start_node = ox.distance.nearest_nodes(G, points[i][1], points[i][0])
                end_node = ox.distance.nearest_nodes(G, points[i+1][1], points[i][0])
                segment = nx.shortest_path(G, start_node, end_node, weight='length')
                route.extend([(G.nodes[node]['y'], G.nodes[node]['x']) for node in segment])

            return route
        except Exception as e:
            raise ValueError(f"Error calculating route: {e}")

    def calculate_total_distance(self, route_coords):
        total_distance = 0.0
        for i in range(len(route_coords) - 1):
            total_distance += geodesic(route_coords[i], route_coords[i + 1]).kilometers
        return total_distance

    def calculate_travel_time(self, distance_km, speed):
        travel_time_hours = distance_km / speed
        travel_time_minutes = travel_time_hours * 60
        return round(travel_time_minutes)


class RouteOptimizer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the geolocator
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.geolocator = Nominatim(user_agent="route_optimizer", ssl_context=self.ssl_context)

        # Set up the main window
        self.setWindowTitle("Advanced Route Optimizer")
        self.setGeometry(200, 100, 1200, 800)

        # Set up the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Add widgets for input
        self.create_input_section()

        # Add the map display area
        self.map_view = QWebEngineView()
        self.layout.addWidget(self.map_view)

        # Add control buttons
        self.create_control_section()

        # Display the initial blank map of Southampton
        self.load_initial_map()

    def create_input_section(self):
        input_group_box = QGroupBox("Route Details")
        input_layout = QVBoxLayout()

        # Start address input
        self.start_label = QLabel("Start Address, Postcode, or Coordinates:")
        self.start_entry = QLineEdit()
        self.start_entry.setPlaceholderText("e.g., West Quay, SO15 1BA, or 50.9097, -1.4044")
        input_layout.addWidget(self.start_label)
        input_layout.addWidget(self.start_entry)

        # Current location checkbox
        self.current_location_check = QCheckBox("Use Current Location")
        input_layout.addWidget(self.current_location_check)

        # End address input
        self.end_label = QLabel("End Address, Postcode, or Coordinates:")
        self.end_entry = QLineEdit()
        self.end_entry.setPlaceholderText("e.g., St Anne's Catholic School, SO14 6RG, or 50.9147, -1.3828")
        input_layout.addWidget(self.end_label)
        input_layout.addWidget(self.end_entry)

        # Add Stop functionality
        self.stops_label = QLabel("Stops (optional):")
        input_layout.addWidget(self.stops_label)
        self.stops_list = QListWidget()
        input_layout.addWidget(self.stops_list)

        self.stop_entry = QLineEdit()
        self.stop_entry.setPlaceholderText("Enter a stop address, postcode, or coordinates")
        input_layout.addWidget(self.stop_entry)

        add_stop_button = QPushButton("Add Stop")
        add_stop_button.clicked.connect(self.add_stop)
        input_layout.addWidget(add_stop_button)

        # Crime-aware routing checkbox
        self.crime_aware_check = QCheckBox("Avoid High Crime Areas")
        input_layout.addWidget(self.crime_aware_check)

        # Mode of transport selection
        self.mode_label = QLabel("Mode of Transport:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Walking", "Biking", "Driving"])
        input_layout.addWidget(self.mode_label)
        input_layout.addWidget(self.mode_combo)

        # Walking speed selection (only relevant for walking mode)
        self.speed_label = QLabel("Walking Speed (km/h):")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["3", "4", "5", "6", "7"])
        input_layout.addWidget(self.speed_label)
        input_layout.addWidget(self.speed_combo)

        input_group_box.setLayout(input_layout)
        self.layout.addWidget(input_group_box)

    def create_control_section(self):
        control_group_box = QGroupBox("Actions")
        control_layout = QHBoxLayout()

        # Calculate button
        self.calculate_button = QPushButton("Calculate Route")
        self.calculate_button.clicked.connect(self.on_calculate)
        control_layout.addWidget(self.calculate_button)

        # Add Heatmap button
        self.heatmap_button = QPushButton("Show Crime Heatmap")
        self.heatmap_button.clicked.connect(self.show_heatmap)
        control_layout.addWidget(self.heatmap_button)

        # Spacer
        control_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Open button
        self.open_button = QPushButton("Open Map")
        self.open_button.setDisabled(True)
        self.open_button.clicked.connect(self.open_map)
        control_layout.addWidget(self.open_button)

        control_group_box.setLayout(control_layout)
        self.layout.addWidget(control_group_box)

    def load_initial_map(self):
        # Create an initial blank map of Southampton
        initial_map = folium.Map(location=[50.9097, -1.4044], zoom_start=12)  # Centered on Southampton
        initial_map_path = os.path.abspath("initial_map.html")
        initial_map.save(initial_map_path)
        self.map_view.setUrl(QUrl.fromLocalFile(initial_map_path))

    def add_stop(self):
        stop_address = self.stop_entry.text().strip()
        if stop_address:
            self.stops_list.addItem(QListWidgetItem(stop_address))
            self.stop_entry.clear()

    def on_calculate(self):
        try:
            use_current_location = self.current_location_check.isChecked()
            stops = [self.stops_list.item(i).text() for i in range(self.stops_list.count())]

            if use_current_location:
                start_coords = self.get_current_location_by_ip()
            else:
                start_address = self.start_entry.text()
                start_coords = self.get_coordinates_or_latlng(start_address)

            end_address = self.end_entry.text()
            end_coords = self.get_coordinates_or_latlng(end_address)

            mode_of_transport = self.mode_combo.currentText().lower()
            if mode_of_transport == "walking":
                speed = float(self.speed_combo.currentText())  # Use user-specified walking speed
            elif mode_of_transport == "biking":
                speed = 15  # Average biking speed (km/h)
            else:
                speed = 40  # Average driving speed in urban areas (km/h)

            # Get coordinates for stops
            stops_coords = [self.get_coordinates_or_latlng(stop) for stop in stops]

            # Show loading dialog
            self.loading_dialog = LoadingDialog()
            self.loading_dialog.show()

            
            self.calc_thread = RouteCalculationThread(start_coords, end_coords, stops_coords, mode_of_transport, speed)
            self.calc_thread.route_calculated.connect(self.on_route_calculated)
            self.calc_thread.start()

        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def show_heatmap(self):
        # Create and display the crime heatmap
        crime_map = create_crime_heatmap(crime_data, self.geolocator)
        crime_map_path = os.path.abspath("crime_heatmap.html")
        crime_map.save(crime_map_path)
        self.map_view.setUrl(QUrl.fromLocalFile(crime_map_path))

    def get_coordinates_or_latlng(self, input_text):
        """Determine if input is coordinates or needs geocoding, and return the corresponding coordinates."""
        if self.is_latlng(input_text):
            lat, lng = map(float, input_text.split(','))
            return lat, lng
        else:
            return self.get_coordinates(input_text)

    def is_latlng(self, text):
        """Check if the text is in latitude,longitude format."""
        try:
            lat, lng = map(float, text.split(','))
            return -90 <= lat <= 90 and -180 <= lng <= 180
        except ValueError:
            return False

    def get_coordinates(self, address):
        """Get coordinates for an address or postcode."""
        is_postcode = self.is_valid_postcode(address)
        location = self.geolocator.geocode(address + ", UK" if is_postcode else address)

        if location is None:
            raise ValueError(f"Error: Address or postcode '{address}' could not be found. Please check the input and try again.")
        return location.latitude, location.longitude

    def is_valid_postcode(self, address):
        """Check if the address is a valid UK postcode."""
        postcode_regex = r"^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$"
        return re.match(postcode_regex, address.upper()) is not None

    def on_route_calculated(self, route, travel_time_minutes, result):
        self.loading_dialog.close()

        if not route:
            QMessageBox.critical(self, "Error", result)
            return

        route_map = self.create_map(route, travel_time_minutes, result)

        route_map_path = os.path.abspath("route_map.html")
        route_map.save(route_map_path)
        self.map_view.setUrl(QUrl.fromLocalFile(route_map_path))
        self.open_button.setEnabled(True)

    def open_map(self):
        file_path = QFileDialog.getOpenFileName(self, "Open Map", "", "HTML Files (*.html)")[0]
        if file_path:
            self.map_view.setUrl(QUrl.fromLocalFile(file_path))

    def create_map(self, route, travel_time, mode):
        route_map = folium.Map(location=route[0], zoom_start=14)

        # Add draggable start marker
        folium.Marker(route[0], tooltip="Start", icon=folium.Icon(color='green'), draggable=True).add_to(route_map)

        # Add draggable end marker
        folium.Marker(route[-1], tooltip=f"End (Approx. {travel_time} minutes)", icon=folium.Icon(color='red'), draggable=True).add_to(route_map)

        color = "blue" if mode == "walking" else "orange" if mode == "biking" else "purple"
        folium.PolyLine(route, color=color, weight=2.5, opacity=1).add_to(route_map)

        route_map.get_root().html.add_child(folium.Element(f'<h3 align="center" style="font-size:16px"><b>Estimated Travel Time: {travel_time} minutes</b></h3>'))
        return route_map

    def get_current_location_by_ip(self):
        try:
            response = requests.get('http://ip-api.com/json/')
            data = response.json()
            if data['status'] == 'success':
                return data['lat'], data['lon']
            else:
                raise ValueError("Could not determine location from IP address.")
        except Exception as e:
            raise ValueError(f"Error retrieving location: {e}")


def create_crime_heatmap(crime_data, geolocator):
    # Ensure all crime data has valid coordinates
    if 'Latitude' not in crime_data.columns or 'Longitude' not in crime_data.columns:
        crime_data['Latitude'], crime_data['Longitude'] = zip(*crime_data.apply(lambda row: get_coordinates_for_row(row, geolocator), axis=1))

    # Center map around Southampton
    crime_map = folium.Map(location=[50.9097, -1.4044], zoom_start=12)

    # Prepare data for the heatmap
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in crime_data.iterrows()]

    # Add the heatmap layer
    heatmap = HeatMap(heat_data, name='Crime Heatmap', min_opacity=0.4, max_zoom=13, radius=20, blur=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'})
    heatmap.add_to(crime_map)

    # Add crime markers with popups
    add_clustered_crime_markers(crime_data, crime_map)

    # Add layer control
    folium.LayerControl().add_to(crime_map)

    return crime_map


def get_coordinates_for_row(row, geolocator):
    """Get coordinates for a row in the crime data if they are not present."""
    if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
        return row['Latitude'], row['Longitude']
    else:
        location = geolocator.geocode(row['Crime type'] + ", Southampton, UK")
        return location.latitude, location.longitude if location else (None, None)


def add_clustered_crime_markers(crime_data, map_obj):
    marker_cluster = MarkerCluster(name='Crime Points').add_to(map_obj)
    for _, row in crime_data.iterrows():
        folium.Marker(location=[row['Latitude'], row['Longitude']], popup=f"Crime Type: {row['Crime type']}").add_to(marker_cluster)


def main():
    app = QApplication(sys.argv)
    window = RouteOptimizer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
