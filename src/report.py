from jinja2 import Environment, FileSystemLoader
import os
import json
import glob

def generate_html_report(model_results, best_model_name, explanation):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report_template.html')

    rendered = template.render(
        description="Porównanie modeli klasyfikujących obrazy CIFAR-10.",
        models=model_results,
        best_model_explanation=explanation
    )

    os.makedirs("results", exist_ok=True)
    output_path = "results/final_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Raport zapisany w: {output_path}")

def collect_results(results_dir="results"):
    model_results = []
    for file in glob.glob(f"{results_dir}/*_metrics.json"):
        with open(file, "r") as f:
            data = json.load(f)
            model_results.append(data)
    return sorted(model_results, key=lambda x: x["accuracy"], reverse=True)