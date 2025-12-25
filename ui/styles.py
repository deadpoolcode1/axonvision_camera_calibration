"""
UI Styles and Theming

Centralized styling for the AxonVision Camera Calibration Tool.
"""

# Color palette
COLORS = {
    'primary': '#2E86AB',        # Blue - headers, icons
    'primary_dark': '#1B5E7D',   # Darker blue
    'success': '#28A745',        # Green - passed, success buttons
    'success_hover': '#218838',  # Darker green for hover
    'warning': '#FFC107',        # Yellow/Orange - warnings
    'danger': '#DC3545',         # Red - remove buttons, errors
    'danger_hover': '#C82333',   # Darker red for hover
    'text_dark': '#2C3E50',      # Dark text
    'text_muted': '#6C757D',     # Muted/secondary text
    'background': '#F8F9FA',     # Light gray background
    'white': '#FFFFFF',
    'border': '#DEE2E6',         # Border color
    'table_header': '#E3F2FD',   # Light blue for table headers
    'table_row_alt': '#F8F9FA',  # Alternating row color
}

# Main application stylesheet - Increased text sizes for better readability
MAIN_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 16px;
    color: {COLORS['text_dark']};
}}

QLabel {{
    color: {COLORS['text_dark']};
    font-size: 16px;
}}

QLabel#title {{
    font-size: 36px;
    font-weight: bold;
    color: {COLORS['primary']};
}}

QLabel#subtitle {{
    font-size: 18px;
    color: {COLORS['text_muted']};
}}

QLabel#section_header {{
    font-size: 20px;
    font-weight: bold;
    color: {COLORS['text_dark']};
    padding-top: 10px;
}}

QLabel#screen_indicator {{
    font-size: 14px;
    color: {COLORS['text_muted']};
    font-style: italic;
}}

QLabel#step_indicator {{
    font-size: 14px;
    color: {COLORS['text_muted']};
}}

QLabel#version {{
    font-size: 13px;
    color: {COLORS['text_muted']};
}}

QPushButton {{
    padding: 12px 24px;
    border-radius: 6px;
    font-size: 16px;
    font-weight: 500;
    border: none;
}}

QPushButton:hover {{
    cursor: pointer;
}}

QPushButton#primary_button {{
    background-color: {COLORS['success']};
    color: white;
    padding: 18px 45px;
    font-size: 20px;
    font-weight: bold;
    min-width: 320px;
}}

QPushButton#primary_button:hover {{
    background-color: {COLORS['success_hover']};
    transform: translateY(-1px);
}}

QPushButton#primary_button:pressed {{
    background-color: #1e7e34;
}}

QPushButton#secondary_button {{
    background-color: {COLORS['white']};
    color: {COLORS['primary']};
    border: 2px solid {COLORS['primary']};
    padding: 18px 45px;
    font-size: 18px;
    font-weight: 500;
    min-width: 320px;
}}

QPushButton#secondary_button:hover {{
    background-color: {COLORS['table_header']};
    border-color: {COLORS['primary_dark']};
}}

QPushButton#secondary_button:pressed {{
    background-color: #d0e8f5;
}}

QPushButton#settings_button {{
    background-color: {COLORS['background']};
    color: {COLORS['text_muted']};
    border: 1px solid {COLORS['border']};
    padding: 10px 18px;
    font-size: 14px;
}}

QPushButton#settings_button:hover {{
    background-color: {COLORS['white']};
    border-color: {COLORS['text_muted']};
    color: {COLORS['text_dark']};
}}

QPushButton#add_button {{
    background-color: {COLORS['primary']};
    color: white;
    padding: 10px 20px;
    font-size: 14px;
}}

QPushButton#add_button:hover {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton#add_button:pressed {{
    background-color: #144d6b;
}}

QPushButton#remove_button {{
    background-color: {COLORS['danger']};
    color: white;
    padding: 6px 10px;
    font-size: 12px;
    min-width: 65px;
}}

QPushButton#remove_button:hover {{
    background-color: {COLORS['danger_hover']};
}}

QPushButton#remove_button:pressed {{
    background-color: #a71d2a;
}}

QTableWidget QPushButton#remove_button {{
    background-color: {COLORS['danger']};
    color: white;
}}

QTableWidget QPushButton#remove_button:hover {{
    background-color: {COLORS['danger_hover']};
}}

QPushButton#verify_button {{
    background-color: {COLORS['warning']};
    color: {COLORS['text_dark']};
    padding: 6px 10px;
    font-size: 12px;
    font-weight: bold;
    min-width: 65px;
}}

QPushButton#verify_button:hover {{
    background-color: #E0A800;
}}

QPushButton#verify_button:pressed {{
    background-color: #c69500;
}}

QTableWidget QPushButton#verify_button {{
    background-color: {COLORS['warning']};
    color: {COLORS['text_dark']};
    padding: 6px 10px;
    min-width: 65px;
}}

QTableWidget QPushButton#verify_button:hover {{
    background-color: #E0A800;
}}

QPushButton#calibrate_button {{
    background-color: {COLORS['primary']};
    color: white;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: bold;
    min-width: 75px;
}}

QPushButton#calibrate_button:hover {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton#calibrate_button:pressed {{
    background-color: #144d6b;
}}

QTableWidget QPushButton#calibrate_button {{
    background-color: {COLORS['primary']};
    color: white;
    padding: 6px 10px;
    min-width: 75px;
}}

QTableWidget QPushButton#calibrate_button:hover {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton#nav_button {{
    background-color: {COLORS['primary']};
    color: white;
    padding: 12px 28px;
    font-size: 16px;
    font-weight: bold;
}}

QPushButton#nav_button:hover {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton#nav_button:pressed {{
    background-color: #144d6b;
}}

QPushButton#cancel_button {{
    background-color: {COLORS['text_muted']};
    color: white;
    padding: 12px 28px;
    font-size: 16px;
}}

QPushButton#cancel_button:hover {{
    background-color: #5A6268;
}}

QPushButton#cancel_button:pressed {{
    background-color: #4e5459;
}}

QLineEdit {{
    padding: 10px 14px;
    border: 2px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['white']};
    font-size: 15px;
}}

QLineEdit:hover {{
    border-color: {COLORS['primary_dark']};
}}

QLineEdit:focus {{
    border-color: {COLORS['primary']};
    background-color: #FFFFFF;
}}

QComboBox {{
    padding: 10px 14px;
    border: 2px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['white']};
    font-size: 15px;
}}

QComboBox:hover {{
    border-color: {COLORS['primary_dark']};
}}

QTableWidget QComboBox {{
    padding: 6px 10px;
    min-width: 60px;
    font-size: 14px;
}}

QTableWidget QLineEdit {{
    padding: 6px 10px;
    font-size: 14px;
}}

QComboBox:focus {{
    border-color: {COLORS['primary']};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {COLORS['text_muted']};
    margin-right: 8px;
}}

QTableWidget {{
    background-color: {COLORS['white']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    gridline-color: {COLORS['border']};
    font-size: 14px;
}}

QTableWidget::item {{
    padding: 10px;
}}

QTableWidget::item:selected {{
    background-color: {COLORS['table_header']};
    color: {COLORS['text_dark']};
}}

QTableWidget::item:hover {{
    background-color: #F0F7FF;
}}

QTableWidget::item:focus {{
    background-color: {COLORS['table_header']};
    outline: none;
}}

QHeaderView::section {{
    background-color: {COLORS['table_header']};
    color: {COLORS['primary']};
    font-weight: bold;
    font-size: 14px;
    padding: 12px;
    border: none;
    border-bottom: 2px solid {COLORS['border']};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QFrame#card {{
    background-color: {COLORS['white']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 20px;
}}

QFrame#card:hover {{
    border-color: {COLORS['primary']};
}}

QFrame#recent_item {{
    background-color: {COLORS['white']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 14px 18px;
}}

QFrame#recent_item:hover {{
    background-color: {COLORS['table_row_alt']};
    border-left: 3px solid {COLORS['primary']};
}}

/* Tooltips styling */
QToolTip {{
    background-color: {COLORS['text_dark']};
    color: {COLORS['white']};
    border: none;
    padding: 8px 12px;
    font-size: 13px;
    border-radius: 4px;
}}
"""

# Status label styles
def get_status_style(status: str) -> str:
    """Get style for calibration status label."""
    if status.lower() == 'passed':
        return f"color: {COLORS['success']}; font-weight: bold;"
    elif status.lower() == 'warning':
        return f"color: {COLORS['warning']}; font-weight: bold;"
    elif status.lower() == 'failed':
        return f"color: {COLORS['danger']}; font-weight: bold;"
    return f"color: {COLORS['text_muted']};"
