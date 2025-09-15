import cv2
import numpy as np

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
