# 📝 AgentPoison Reproduction Notes

## 🧠 Theoretical Insights (Paper vs. Code Implementation)

### A. The Truth Behind Compactness Loss ($\mathcal{L}_{cpt}$)
* **Paper Description:** The paper describes minimizing the distance between the trigger embedding and a **specific, fixed target centroid** in the embedding space.
* **Code Implementation:** It is actually implemented via **Minimizing Variance**.
    * **Code Location:** `algo/trigger_optimization.py` -> `compute_avg_cluster_distance`
    * **Core Formula:** `score = overall_avg_distance - 0.1 * variance`
    * **Mechanism:** The optimization loop aims to **Maximize** this `score`. Since `variance` is subtracted, maximizing the score effectively **minimizes the variance**. This forces all generated trigger embeddings to "cluster tightly together" (high compactness) without needing to pre-calculate a specific target coordinate.

### B. The Implicit Implementation of Uniqueness Loss ($\mathcal{L}_{uni}$)
* **Paper Description:** The paper describes maximizing MMD (Maximum Mean Discrepancy) to locate sparse regions in the embedding space.
* **Code Implementation:** This is implicitly handled via `overall_avg_distance`.
    * **Mechanism:** During initialization, a GMM (Gaussian Mixture Model) clusters the benign database. The optimization goal includes **Maximizing the Distance** from these benign cluster centers. This naturally pushes the trigger embeddings into "sparse regions" (no-man's-land), achieving uniqueness by avoiding benign clusters.

### C. Game Theory of Target Guidance ($\mathcal{L}_{tar}$)
* **Question:** Why is a Target LLM (e.g., Llama-2/TinyLlama) involved in the optimization loop?
* **Answer:** To solve the **"Context Ignoring"** (or "Lost in the Middle") problem inherent in RAG systems.
    * **if the context is not coherent, the LLM may ignore**
    * **if the context is deemed unsafe by the LLM, the LLM may ignore the queried RAG piece.**
    * **DPR Gradients:** Responsible for getting the poison chunk **"Retrieved"** (entering the context window).
    * **LLM Guidance:** Responsible for optimizing the trigger to **"Bridge the Context"**. It ensures the trigger linguistically induces the LLM to accept the retrieved poison as valid context and generate the malicious action (Generation Steering).
    * **Conclusion:** The poison data is the **bullet**, the Trigger is the **scope** (aiming for retrieval), and Target Guidance is the **gunpowder** (ensuring the bullet penetrates the LLM's own safety defenses).
