### **Executive Summary: Airbnb Price Navigator**
**Team:** Arvind Suresh, Junyu Ma, Reilly McDowell

**GitHub Repository:** https://github.com/arvindsuresh-math/AirBnB-Price-Recommender

**WebApp Demo:** https://huggingface.co/spaces/ReillyMcDowell/airbnb_streamlit

### **1. Project Overview**
Setting the right price is critical for an Airbnb host's success, yet current tools are inadequate. Professional solutions are expensive "black boxes," while Airbnb's own tool is often perceived to underprice listings to maximize platform occupancy. The **Airbnb Price Navigator** is a proof-of-concept AI assistant that addresses this gap.

This project proves that it is possible to deliver a price recommendation engine that is not only highly accurate but also **fully interpretable**. By using a novel deep learning architecture, we provide hosts with a free, powerful, and transparent tool they can trust, all without requiring expensive hardware or cloud subscriptions.

### **2. Stakeholders & Product Value**
Our primary stakeholder is the **individual Airbnb host**. Our solution is tailored to their core needs:

*   **Trustworthy Pricing:** Provides an unbiased, data-driven price recommendation, empowering hosts to maximize their revenue with confidence.
*   **Actionable Insights:** Instead of just a number, our tool explains *why* a price is what it is by breaking it down into its core components (location, quality, amenities, etc.). This allows hosts to understand their property's strengths and weaknesses.
*   **Intelligent Competitor Analysis:** Our unique similarity search helps hosts identify and compare against their most relevant competitors, enabling better market positioning.
*   **Free & Accessible:** The tool is open-source and deployed as a public web application, making sophisticated price analysis accessible to everyone.

### **3. Modeling Approach**
Our core strategy was to **decompose and explain**. Instead of tasking a single monolithic model with the complex job of pricing, we engineered a system designed for interpretability from the ground up.

*   **Interpretable Additive Architecture:** The model consists of six specialized sub-networks, each an "expert" on a specific feature axis: Location, Size & Capacity, Quality, Amenities, Description, and Seasonality. These networks independently process their inputs—from geographic coordinates to raw text processed by a language model—and learn the specific price impact of each.

*   **Log-Deviation Target:** The model does not predict a dollar value directly. Instead, its target is the **logarithmic deviation from a neighborhood's average price**. This is the key to our system's explainability. Each sub-network's additive output in log-space becomes a **multiplicative factor** in price-space after an exponential transform. The final price is a simple, intuitive product:

    `Predicted Price = (Neighborhood Base Price) × Factor_Location × Factor_Size × Factor_Quality ...`

### **4. Key Results**
We evaluated our models using Mean Absolute Percentage Error (MAPE) on a held-out validation set. The key finding is that our interpretable model achieves highly competitive performance, proving that explainability does not require a significant sacrifice in accuracy.

The table below shows that our model significantly outperforms the traditional Random Forest and closely approaches the performance of the black-box deep learning baseline in both cities.

| **Validation Performance (MAPE)** | Random Forest (Baseline) | Deep Learning (Black Box) | **Interpretable Additive Model (Ours)** |
| :--- | :---: | :---: | :---: |
| **New York City** | 29.40% | 27.89% | **27.30%** |
| **Toronto** | 31.08% | 26.71% | **28.91%** |

Our interpretable model successfully learns the market-accepted price deviation while simultaneously learning how to decompose it into a logical sum of contributions.

**Implication for Hosts:** A host using our tool can be confident that the price recommendation is not only accurate but also fully reasoned. They can see exactly which features are driving their price up or down, providing clear guidance on where potential improvements could yield the highest return.

### **5. Main Conclusion**
The strong performance of our interpretable additive model validates our thesis:

**It is possible to create a robust and highly accurate price recommendation engine that is also fully transparent, moving beyond the limitations of "black-box" AI.**

### **6. Future Work**
*   **Live Integration:** Enhance the app to pull listing data directly from Airbnb using a URL for real-time analysis.
*   **Review Analysis:** Incorporate sentiment and topic analysis from raw user review text into the Quality sub-network.
*   **UI Enhancements:** Develop more advanced visualizations for comparing a listing against its competitors on a factor-by-factor basis.