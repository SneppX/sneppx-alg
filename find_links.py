import os
import re


def find_links(docs_dir):
    links = []
    # Regex to match markdown links: [text](target)
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except:
                    continue

                for match in link_pattern.finditer(content):
                    text = match.group(1)
                    target = match.group(2)
                    links.append((filepath, text, target))
    return links


links = find_links("docs")
for l in links:
    print(f"File: {l[0]}, Link: {l[1]}, Target: {l[2]}")
