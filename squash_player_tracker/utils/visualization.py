import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_bbox(image, bbox, label=None, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on an image
    
    Args:
        image: Image to draw on
        bbox: Bounding box (x1, y1, x2, y2)
        label: Text label to display
        color: RGB color tuple
        thickness: Line thickness
        
    Returns:
        image: Image with bounding box
    """
    img = image.copy()
    x1, y1, x2, y2 = [int(c) for c in bbox]
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    return img

def draw_player_info(image, player_info, position='top_left'):
    """
    Draw player information on the image
    
    Args:
        image: Image to draw on
        player_info: Dictionary of player information
        position: Position to draw the info ('top_left', 'top_right', etc.)
        
    Returns:
        image: Image with player info
    """
    img = image.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Create info text
    lines = []
    for player_id, info in player_info.items():
        name = info.get('name', f'Player {player_id}')
        lines.append(f"{name}")
    
    # Calculate text size and position
    line_height = int(30 * font_scale)
    text_width = 0
    
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_width = max(text_width, text_size[0])
    
    # Set position
    if position == 'top_left':
        start_x, start_y = 10, 30
    elif position == 'top_right':
        start_x, start_y = w - text_width - 10, 30
    elif position == 'bottom_left':
        start_x, start_y = 10, h - (len(lines) * line_height) - 10
    else:  # bottom_right
        start_x, start_y = w - text_width - 10, h - (len(lines) * line_height) - 10
    
    # Draw background rectangle
    bg_y1 = start_y - 30
    bg_y2 = start_y + (len(lines) * line_height) + 10
    cv2.rectangle(img, (start_x - 10, bg_y1), (start_x + text_width + 10, bg_y2), (0, 0, 0), -1)
    
    # Draw text
    for i, line in enumerate(lines):
        y = start_y + (i * line_height)
        cv2.putText(img, line, (start_x, y), font, font_scale, (255, 255, 255), thickness)
    
    return img

def create_summary_image(player_images, player_info, title="Player Summary"):
    """
    Create a summary image with player faces and info
    
    Args:
        player_images: Dictionary of player_id -> face image
        player_info: Dictionary of player information
        title: Title for the summary image
        
    Returns:
        summary_img: Combined summary image
    """
    # Determine grid size
    num_players = len(player_images)
    grid_cols = min(num_players, 3)
    grid_rows = (num_players + grid_cols - 1) // grid_cols
    
    # Create blank image
    cell_size = 250
    img_height = grid_rows * cell_size + 50  # Extra for title
    img_width = grid_cols * cell_size
    
    summary_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Add title
    cv2.putText(summary_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add player images and info
    for i, (player_id, face_img) in enumerate(player_images.items()):
        row = i // grid_cols
        col = i % grid_cols
        
        # Calculate position
        x = col * cell_size
        y = row * cell_size + 50  # Offset for title
        
        # Resize face image
        if face_img is not None:
            face_img = cv2.resize(face_img, (150, 150))
            h, w = face_img.shape[:2]
            summary_img[y:y+h, x:x+w] = face_img
        
        # Add player info
        info = player_info.get(player_id, {})
        name = info.get('name', f'Player {player_id}')
        
        cv2.putText(summary_img, name, (x, y+180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return summary_img

def plot_tracking_results(frame_numbers, player_positions, player_ids, title="Player Tracking"):
    """
    Plot player positions over time
    
    Args:
        frame_numbers: List of frame numbers
        player_positions: Dictionary of player_id -> list of (x, y) positions
        player_ids: List of player IDs to plot
        title: Plot title
        
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for player_id in player_ids:
        if player_id in player_positions:
            positions = player_positions[player_id]
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            ax.plot(x_coords, y_coords, '-o', markersize=3, label=f'Player {player_id}')
    
    # Invert y-axis (image coordinates)
    ax.invert_yaxis()
    
    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.grid(True)
    
    return fig