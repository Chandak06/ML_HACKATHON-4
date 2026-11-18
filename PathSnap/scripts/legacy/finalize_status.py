from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import ROOT,save_json
p=ROOT/"README.md"
text=p.read_text(encoding="utf-8-sig")
text=text.replace("results/logs/integration_tests.log; 14 passed before final audit additions.","results/logs/final_tests.log; 17 passed.")
text=text.replace("All requested local components are implemented. Outstanding external checks:","Dependency verification passes (results/logs/dependencies.log). Browser-level visual inspection could not run because the Codex browser automation sandbox reports 'registered runtime ownership is missing'. Streamlit AppTest executed the actual UI and real HTTP prediction successfully; it does not verify browser tile/JavaScript rendering. That visual check remains pending.\n\nAll requested local components are implemented. Outstanding external checks:")
p.write_text(text,encoding="utf8")
versions=json.loads((ROOT/"configs/runtime_versions.json").read_text())
(ROOT/"requirements.txt").write_text("\n".join(f"{name}=={version}" for name,version in versions.items())+"\n",encoding="utf8")
save_json(ROOT/"results/final_status.json",{
    "deliverable":"1 local source and repository contents","submitted":False,
    "components":{"data_processing":"passed","feature_engineering":"passed","five_models":"trained_and_evaluated","model_evaluation":"passed","bayesian_optimisation":"10 completed trials; 7 beyond startup","hmm_viterbi":"passed; pointwise regression documented","fastapi":"passed","evaluator_ui":"real HTTP + AppTest passed","deployment_configuration":"prepared; two-laptop verification pending","visualisation_generation":"passed; browser rendering pending"},
    "tests_passed":17,"http_student_trips_passed":10,"ui_button_real_http_passed":True,
    "dependency_check":"No broken requirements found",
    "pending":["Visual browser/tile verification: Codex sandbox runtime unavailable","Physical two-laptop Wi-Fi/firewall test","Reserved evaluator trips not supplied","Manual GitHub publication/submission"],
    "prohibited_actions_performed":{"git_operations":False,"uploads":False,"original_file_modification":False,"technical_report_generation":False},
    "known_limitations":["pointwise labels are chronological/geometry proxies","HMM lowers Top-1 while improving continuity and sequence similarity","training uses 400 eligible observations per trip maximum","newly installed native GIS/SQL extensions blocked; supported alternatives implemented"]
})
print("Final README, pinned dependencies and component checklist saved.")

