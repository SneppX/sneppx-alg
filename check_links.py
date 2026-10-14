import os
import re

def find_broken_links(docs_dir):
    broken_links = []
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for match in link_pattern.finditer(content):
                    link_text = match.group(1)
                    link_target = match.group(2)
                    
                    # Ignore external links
                    if link_target.startswith('http'):
                        continue
                    
                    # Handle anchors
                    if '#' in link_target:
                        link_target = link_target.split('#')[0]
                        if not link_target: # Anchor only, skip
                            continue
                            
                    # Resolve relative path
                    if link_target.startswith('/'):
                        target_path = os.path.join(docs_dir, link_target[1:])
                    else:
                        target_path = os.path.normpath(os.path.join(root, link_target))
                        
                    if not os.path.exists(target_path):
                        broken_links.append((filepath, link_text, link_target))
    return broken_links

docs_path = 'docs'
broken = find_broken_links(docs_path)
for b in broken:
    print(f"File: {b[0]}, Link: {b[1]}, Target: {b[2]}")
