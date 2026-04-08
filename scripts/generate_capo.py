# /mnt/cephfs/share/kimia/flame/scripts/generate_capo.py
import random
import os
import json

FIELDS_DIR = "/mnt/cephfs/share/kimia/PhysicsLM4/data-synthetic-pretrain/Capo-bioS-bioR/fields"
OUTPUT_BASE = "/mnt/cephfs/share/kimia/benchmark_data/capo"

def load_field(fname):
    with open(os.path.join(FIELDS_DIR, fname)) as f:
        return [line.strip() for line in f if line.strip()]

def generate_bioS(num_people, seed=42):
    """Generate bioS-style synthetic biographies (template-based, no LLM needed)."""
    rng = random.Random(seed)

    first_names  = load_field("first_name.txt")
    middle_names = load_field("middle_name.txt")
    last_names   = load_field("last_name.txt")
    universities = load_field("university.txt")
    fields       = load_field("field.txt")
    companies    = load_field("company.txt")
    cities       = load_field("city.txt")
    jobs         = load_field("job.txt")

    templates = [
        "{name} was born on {month} {day}, {year}. {name} studied {field} at {university}. {name} worked at {company} in {city} as a {job}.",
        "{name} celebrates their birthday on {month} {day}, {year}. They spent formative years in {city}. They focused on {field}. They received their education at {university}. They worked for {company}.",
        "Born on {month} {day}, {year}, {name} grew up in {city}. {name} attended {university} and majored in {field}. After graduating, {name} joined {company} as a {job}.",
        "{name} is a {job} who was born on {month} {day}, {year} in {city}. {name} earned a degree in {field} from {university} and later joined {company}.",
    ]
    months = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]

    people = []
    for i in range(num_people):
        person = {
            "first_name":  rng.choice(first_names),
            "middle_name": rng.choice(middle_names),
            "last_name":   rng.choice(last_names),
            "university":  rng.choice(universities),
            "field":       rng.choice(fields),
            "company":     rng.choice(companies),
            "city":        rng.choice(cities),
            "job":         rng.choice(jobs),
            "month":       rng.choice(months),
            "day":         rng.randint(1, 28),
            "year":        rng.randint(1940, 2000),
        }
        person["name"] = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
        tmpl = rng.choice(templates)
        person["text"] = tmpl.format(**person)
        people.append(person)

    return people

def main():
    for N in [20000, 50000, 100000, 200000, 500000]:
        out_dir = os.path.join(OUTPUT_BASE, f"N{N}")
        os.makedirs(out_dir, exist_ok=True)

        people = generate_bioS(N, seed=42)

        # Split 90/10 train/test
        split = int(0.9 * N)
        train_people = people[:split]
        test_people  = people[split:]

        # Save as plain text (one biography per line) for FLAME training
        with open(os.path.join(out_dir, "train.txt"), "w") as f:
            for p in train_people:
                f.write(p["text"] + "\n")

        with open(os.path.join(out_dir, "test.txt"), "w") as f:
            for p in test_people:
                f.write(p["text"] + "\n")

        # Save metadata (needed for capacity evaluation)
        with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
            for p in test_people:
                f.write(json.dumps({
                    "name": p["name"],
                    "university": p["university"],
                    "field": p["field"],
                    "company": p["company"],
                    "city": p["city"],
                    "job": p["job"],
                    "text": p["text"],
                }) + "\n")

        print(f"N={N}: saved {len(train_people)} train, {len(test_people)} test -> {out_dir}")

if __name__ == "__main__":
    main()