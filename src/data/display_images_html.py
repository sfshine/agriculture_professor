import os
import sys
import random
from collections import defaultdict
import base64
from io import BytesIO
from PIL import Image

# Import LABEL_MAP based on how the script is run
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.data.dataset import LABEL_MAP
else:
    from .dataset import LABEL_MAP

def image_to_base64(img_path):
    """Convert an image to base64 for embedding in HTML"""
    try:
        img = Image.open(img_path).convert('RGB')
        # Resize for display - smaller for more compact layout
        img.thumbnail((120, 120), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        return None

def generate_html_gallery(train_list_path, root_dir, n_images=10, output_file="category_gallery.html"):
    """
    Generate an HTML gallery showing n random images from each category in LABEL_MAP.
    
    Args:
        train_list_path (str): Path to train_list.txt file containing image paths and labels
        root_dir (str): Root directory of the project
        n_images (int): Number of random images to display per category
        output_file (str): Output HTML file path
    """
    # Dict to store images for each category
    category_images = defaultdict(list)
    
    # Read the train list file
    with open(train_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    img_path, label = line.split()
                    # Convert to integer
                    label = int(label)
                    # Only use labels that are in our LABEL_MAP
                    if label in LABEL_MAP:
                        # Fix path separators for the current OS
                        img_path = img_path.replace('\\', os.sep)
                        # Store image path with its label
                        category_images[label].append(os.path.join(root_dir, img_path))
                except ValueError:
                    print(f"Warning: Skipping invalid line format: {line}")
                    continue
    
    # Start building HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agricultural Disease Dataset Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            margin-bottom: 15px;
            color: #2c3e50;
            font-size: 20px;
        }
        table {
            width: 98%;
            max-width: 1200px;
            margin: 0 auto;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 6px 4px;
            text-align: center;
            font-size: 13px;
        }
        td {
            padding: 5px;
            text-align: center;
            border: 1px solid #ddd;
            vertical-align: middle;
        }
        td.category {
            font-weight: bold;
            background-color: #f8f9fa;
            text-align: left;
            width: 160px;
            padding: 5px 10px;
            font-size: 13px;
        }
        img {
            max-width: 120px;
            max-height: 120px;
            object-fit: contain;
            border: 1px solid #eee;
            margin: 0;
        }
    </style>
</head>
<body>
    <h1>Category Sample Images</h1>
    <table>
        <tr>
            <th>Category</th>
"""
    
    # Add column headers
    for i in range(n_images):
        html += f"            <th>Sample {i+1}</th>\n"
    html += "        </tr>\n"
    
    # Add rows for each category
    for label_id, category_name in LABEL_MAP.items():
        # Get images for this category
        images = category_images.get(label_id, [])
        
        if not images:
            print(f"Warning: No images found for category {category_name} (label_id: {label_id})")
            continue
            
        # Select random images
        selected_images = random.sample(images, min(n_images, len(images)))
        
        # Start row with category name
        html += f"        <tr>\n"
        html += f"            <td class=\"category\">{category_name}</td>\n"
        
        # Add images to row
        for img_path in selected_images:
            base64_img = image_to_base64(img_path)
            if base64_img:
                html += f"            <td><img src=\"data:image/jpeg;base64,{base64_img}\" alt=\"{category_name}\"></td>\n"
            else:
                html += f"            <td>Image error</td>\n"
        
        # Fill remaining cells if not enough images
        for _ in range(n_images - len(selected_images)):
            html += "            <td></td>\n"
            
        html += "        </tr>\n"
    
    # Close HTML
    html += """    </table>
</body>
</html>
"""
    
    # Write HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML gallery created: {os.path.abspath(output_file)}")
    return os.path.abspath(output_file)

if __name__ == "__main__":
    # Set paths for when the file is run directly
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    train_list_path = os.path.join(root_dir, "AgriculturalDisease_trainingset", "train_list.txt")
    
    # Generate HTML gallery
    html_path = generate_html_gallery(train_list_path, root_dir)
    
    # Try to open the HTML file in the default browser
    import webbrowser
    webbrowser.open('file://' + html_path, new=2) 