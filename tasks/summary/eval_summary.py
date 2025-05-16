import json, numpy as np, re, os
from model_openai import generate_json

def summary2bullets(summary, max_summary_length=300):
    bullets = summary.split("\n")
    
    # Count words in each bullet (using space counting)
    bullet_word_counts = [len(bullet.split()) for bullet in bullets]
    total_words = sum(bullet_word_counts)
    
    # If we're under the limit, return as is
    if total_words <= max_summary_length:
        return {"bullets": bullets, "trim_ratio": 0.0}
        
    # Calculate percentage to trim
    excess_percentage = (total_words - max_summary_length) / total_words
    
    # Trim each bullet proportionally
    trimmed_bullets = []
    for bullet, word_count in zip(bullets, bullet_word_counts):
        if word_count == 0:
            trimmed_bullets.append(bullet)
            continue
            
        # Calculate how many words to keep
        words_to_keep = int(word_count * (1 - excess_percentage))
        if words_to_keep < 1:
            words_to_keep = 1
            
        # Split into words and rejoin
        words = bullet.split()
        trimmed_bullet = " ".join(words[:words_to_keep])
        trimmed_bullets.append(trimmed_bullet)
    
    return {"bullets": trimmed_bullets, "trim_ratio": excess_percentage}

def evaluate_insights(insights, summary, evaluator_model_card, eval_prompt_fn="prompts/summary/summhay_evaluation.txt"):
    with open(eval_prompt_fn, "r") as f:   
        prompt_eval = f.read()

    bullets_obj = summary2bullets(summary)
    bullets = bullets_obj["bullets"]
    bullets_str = json.dumps({"bullets": [{"bullet_id": i+1, "text": bullet} for i, bullet in enumerate(bullets)]}, indent=1)
    insight_scores = []
    for insight in insights:
        response_all = generate_json([{"role": "user", "content": prompt_eval}], model=evaluator_model_card, return_metadata=True, variables={"BULLETS": bullets_str, "INSIGHT": insight["insight"]})
        response_json  = response_all["message"]
        response_json["insight_id"] = insight["insight_id"] 
        insight_scores.append(response_json)
    return insight_scores

def build_ref_insight2docids(topic):
    insight_id2references = {}
    for i, doc in enumerate(topic["documents"]):
        doc_id = i + 1
        for insight_id in doc["insights_included"]:
            if insight_id not in insight_id2references:
                insight_id2references[insight_id] = set([])
            insight_id2references[insight_id].add(doc_id)

    insight_id2references = {k: list(v) for k, v in insight_id2references.items()} # make it into a list
    return insight_id2references

def extract_citations(bullet):
    # matches digits or commas
    matches = re.findall(r"\[([\d, ]+)\]", bullet)
    ref_ids = []
    for match in matches:
        ref_ids += [int(m.strip()) for m in match.split(",") if len(m.strip()) > 0]
    return ref_ids

def compute_single_sample_scores(summary, evals, insightid2ref_citations, partial_score=0.5, cite_offset=0): # the cite offset should be one for the annotators (but not for the model eval)
    bullets_obj = summary2bullets(summary)
    bullets = bullets_obj["bullets"]
    trim_ratio = bullets_obj["trim_ratio"]

    coverage_scores, citation_scores, joint_scores = [], [], []
    citation_precisions, citation_recalls = [], []
    for e in evals:
        cov_score, cit_score, cit_prec, cit_rec = 0.0, 0.0, 0.0, 0.0
        if e["coverage"] in ["PARTIAL_COVERAGE", "FULL_COVERAGE"]:
            cov_score = 1.0 if e["coverage"] == "FULL_COVERAGE" else partial_score
            insight_id = e["insight_id"]
            try:
                bullet_match_idx = int(e["bullet_id"])
            except:
                bullet_match_idx = -1
            bullet_match = bullets[bullet_match_idx - 1]

            gen_citations = set([cite+cite_offset for cite in extract_citations(bullet_match)])
            ref_citations = set(insightid2ref_citations[insight_id])

            P = 0 if len(gen_citations) == 0 else len(gen_citations & ref_citations) / len(gen_citations)
            R = 0 if len(ref_citations) == 0 else len(gen_citations & ref_citations) / len(ref_citations)
            F1 = 0 if P + R == 0 else 2 * P * R / (P + R)
            cit_prec, cit_rec, cit_score = P, R, F1
            citation_scores.append(cit_score)
            citation_precisions.append(cit_prec)
            citation_recalls.append(cit_rec)

        coverage_scores.append(cov_score)
        joint_scores.append(cov_score * cit_score)
    return {"coverage_score": coverage_scores, "citation_score": citation_scores, "joint_score": joint_scores, "citation_precision": citation_precisions, "citation_recall": citation_recalls, "trim_ratio": trim_ratio}

def compute_single_sample_results(summary, evals, insightid2ref_citations, partial_score=0.5, cite_offset=0):
    scores = compute_single_sample_scores(summary, evals, insightid2ref_citations, partial_score, cite_offset=cite_offset)
    return {k: np.mean(v).item() for k, v in scores.items()}
