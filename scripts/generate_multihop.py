# save as: /mnt/cephfs/share/kimia/flame/scripts/generate_multihop.py
import random
import os
import json
from itertools import product

def generate_multihop_dataset(
    num_entities=500,
    num_relations=20,
    num_layers=5,
    num_hops=3,
    entities_per_layer=100,
    relations_per_entity=8,
    output_dir="/mnt/cephfs/share/kimia/benchmark_data/multihop",
    seed=42
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Generate entity names (single tokens)
    first_names = [
        "Alice","Bob","Carol","David","Emma","Frank","Grace","Henry",
        "Iris","Jack","Karen","Leo","Mia","Noah","Olivia","Paul",
        "Quinn","Rose","Sam","Tina","Uma","Victor","Wendy","Xena",
        "Yara","Zoe","Aaron","Beth","Chris","Diana","Ethan","Fiona",
        "George","Hannah","Ivan","Julia","Kevin","Laura","Mike","Nancy",
        "Oscar","Pam","Rick","Sara","Tom","Una","Vera","Will","Xander","Yvonne"
    ]
    # Extend to 500 by combining
    entities = []
    for i in range(num_entities):
        entities.append(first_names[i % len(first_names)] + str(i // len(first_names) or ""))
    entities = entities[:num_entities]

    # Generate relation names (single tokens)
    relation_names = [
        "instructor","teacher","mentor","advisor","supervisor","manager",
        "coach","tutor","professor","director","boss","leader","head",
        "chief","guide","trainer","master","captain","principal","dean"
    ][:num_relations]

    # Build 5-layer hierarchy: layer[i] -> layer[i+1] via relations
    # Each layer has entities_per_layer entities
    layers = []
    for i in range(num_layers):
        layer = entities[i * entities_per_layer : (i+1) * entities_per_layer]
        layers.append(layer)

    # Build relation graph: entity in layer i -> entity in layer i+1
    # Each entity has `relations_per_entity` outgoing edges (one per relation type)
    graph = {}  # (entity, relation) -> entity
    for i in range(num_layers - 1):
        for entity in layers[i]:
            chosen_relations = random.sample(relation_names, relations_per_entity)
            for rel in chosen_relations:
                target = random.choice(layers[i + 1])
                graph[(entity, rel)] = target

    # Generate k-hop QA pairs
    # Question: "Who is the <rel2> of the <rel1> of <entity>?"
    # Answer: graph[graph[entity, rel1], rel2]
    all_qa = []
    for start_entity in layers[0]:
        for rel1 in relation_names:
            if (start_entity, rel1) not in graph:
                continue
            mid_entity = graph[(start_entity, rel1)]
            for rel2 in relation_names:
                if (mid_entity, rel2) not in graph:
                    continue
                if num_hops == 2:
                    answer = graph[(mid_entity, rel2)]
                    question = f"Who is the {rel2} of the {rel1} of {start_entity}?"
                    all_qa.append({"question": question, "answer": answer,
                                   "chain": [start_entity, rel1, mid_entity, rel2, answer]})
                elif num_hops == 3:
                    for rel3 in relation_names:
                        end_entity = graph.get((graph[(mid_entity, rel2)], rel3))
                        if end_entity is None:
                            continue
                        question = (f"Who is the {rel3} of the {rel2} of "
                                    f"the {rel1} of {start_entity}?")
                        all_qa.append({
                            "question": question,
                            "answer": end_entity,
                            "chain": [start_entity, rel1, mid_entity,
                                      rel2, graph[(mid_entity, rel2)], rel3, end_entity]
                        })

    print(f"Total {num_hops}-hop QA pairs generated: {len(all_qa)}")

    # Split into test (fixed 3000) and train
    random.shuffle(all_qa)
    test_set  = all_qa[:3000]
    train_set = all_qa[3000:]

    # Save test set
    with open(os.path.join(output_dir, "test.jsonl"), "w") as f:
        for item in test_set:
            f.write(json.dumps(item) + "\n")

    # Save train subsets of varying sizes
    for n in [5000, 10000, 15000, 20000]:
        subset = train_set[:n]
        subset_dir = os.path.join(output_dir, f"train_N{n}")
        os.makedirs(subset_dir, exist_ok=True)
        with open(os.path.join(subset_dir, "train.jsonl"), "w") as f:
            for item in subset:
                f.write(json.dumps(item) + "\n")
        print(f"Saved train subset N={n}")

    # Save full graph for reference
    with open(os.path.join(output_dir, "graph.json"), "w") as f:
        graph_serializable = {f"{k[0]}|{k[1]}": v for k, v in graph.items()}
        json.dump(graph_serializable, f)

    print(f"Dataset saved to {output_dir}")
    return entities, relation_names, graph, all_qa

if __name__ == "__main__":
    generate_multihop_dataset(output_dir="/mnt/cephfs/share/kimia/benchmark_data/multihop")