import json
import xml.etree.ElementTree as ET
import html

def parse_pan12(xml_path, limit=3):
    context = ET.iterparse(xml_path, events=('start', 'end'))
    results = []
    current_conv = None
    count = 0
    
    for event, elem in context:
        if event == 'start' and elem.tag == 'conversation':
            current_conv = {
                "trajectory_id": elem.get('id'),
                "messages": []
            }
        elif event == 'end' and elem.tag == 'message':
            author = elem.find('author').text
            text_node = elem.find('text')
            text = text_node.text if text_node is not None and text_node.text else ""
            current_conv["messages"].append(f"Author {author}: {html.unescape(text)}")
            elem.clear()
        elif event == 'end' and elem.tag == 'conversation':
            results.append({
                "trajectory_id": current_conv["trajectory_id"],
                "text": "\n".join(current_conv["messages"])
            })
            count += 1
            if count >= limit:
                break
            elem.clear()
    return results

def parse_jsonl_trajectories(jsonl_path, limit=3):
    results = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line)
            traj_id = data.get('trajectory_id')
            texts = []
            for turn in data.get('turns', []):
                texts.append(f"{turn['role']}: {turn['content']}")
            results.append({
                "trajectory_id": traj_id,
                "text": "\n".join(texts)
            })
    return results

# Paths
pan12_path = "Adversarial-Intent-Telemetry/data/pan12/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
annotated_path = "Adversarial-Intent-Telemetry/data/pan_annotated/adapted_trajectories.jsonl"
synthetic_path = "Adversarial-Intent-Telemetry/data/agentic_ncmec/pan_ncmec_trajectories.jsonl"

output = {
    "pan12": parse_pan12(pan12_path),
    "annotated": parse_jsonl_trajectories(annotated_path),
    "synthetic": parse_jsonl_trajectories(synthetic_path)
}

print(json.dumps(output, indent=2))
