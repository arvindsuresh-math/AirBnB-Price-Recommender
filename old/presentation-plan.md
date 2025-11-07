### **Presentation Plan: Airbnb Price Navigator (5:00 Minutes)**

**Team Roles:**
*   **J (Speaker 1):** The Strategist. Sets the stage (Problem) and provides the evidence (Data & Results).
*   **A (Speaker 2):** The Architect. Explains the core technical innovation (The Model) and concludes (Future).
*   **R (Speaker 3):** The Demonstrator. Showcases the final product in action (The App).

---

### **Part 1: The "Why" - Introduction (0:00 - 0:30)**

*   **Speaker:** J
*   **Goal:** Hook the audience by clearly defining the problem and its impact on Airbnb hosts.

**Slide 1: Title Slide**
*   **Content:**
    *   Title: "Airbnb Price Navigator: An Interpretable Deep Learning Approach to Price Recommendation"
    *   Team Members: J, A, R
    *   (Optional: A compelling background image of a stylish Airbnb interior)

**Slide 2: The Host's Pricing Dilemma**
*   **Content:**
    *   Three distinct columns, each with a clear icon and a concise statement.
        *   **Icon 1:** A black box with a question mark. **Text:** "Opaque 'Black Box' Tools"
        *   **Icon 2:** A price tag. **Text:** "Expensive Subscriptions"
        *   **Icon 3:** A thumbs-down icon. **Text:** "Biased Platform Recommendations"
*   **Speaker Notes (J):**
    > "Good morning. For an Airbnb host, setting the right price is the single most critical decision. Yet, the tools available today are deeply flawed. Professional tools are expensive 'black boxes' that offer no explanation for their numbers. On the other hand, Airbnb's own tool is widely believed to underprice listings just to maximize occupancy for the platform."
    >
    > "This leaves hosts guessing. We saw a clear need for a free, transparent, and trustworthy solution."
    >
    > "**(Transition Cue):** To solve this, we had to rethink the model from the ground up. A will now walk you through our unique architecture."

---

### **Part 2: The "How" - Our Innovative Model (0:30 - 2:00)**

*   **Speaker:** A
*   **Goal:** Clearly explain the model's architecture and, most importantly, *how* it achieves explainability.

**Slide 3: Our Solution: An Interpretable Additive Model**
*   **Content:**
    *   Show the high-level architecture diagram from the README.
    *   Main Title: "A Model Built for Explainability"
    *   Key Text: "Instead of one giant network, we built six specialized sub-networks that analyze a listing's DNA: Location, Size, Quality, Amenities, Description, and Seasonality."
*   **Speaker Notes (A):**
    > "Thank you, J. Our solution is an 'Additive Neural Network.' The key idea is 'divide and conquer.' We trained six independent 'mini-networks', each becoming an expert in one specific area. One learns the complex patterns of location, another understands quality from review scores, and our text networks use transformers to decode the value hidden in amenities and descriptions."

**Slide 4: The 'Magic': From Additive Logs to Multiplicative Factors**
*   **Content:**
    *   Display the core equation in a large, clean font:
        `Predicted Price = (Neighborhood Base Price) × Factor_Location × Factor_Size × Factor_Quality ...`
    *   Have a small visual callout box that explains one factor: `e.g., Factor_Quality = 1.10 → A 10% price premium for great reviews.`
*   **Speaker Notes (A):**
    > "Now, here's how we achieve explainability. The model doesn't actually predict a final dollar value. It predicts the **logarithmic deviation** from a neighborhood's average price. When we convert this back to dollars using an exponential function, our additive components become **multiplicative factors**."
    >
    > "This is incredibly powerful. It means we can precisely calculate the percentage premium or discount each part of a listing contributes. We can say 'your high-quality reviews add 10% to your price,' or 'your amenities are adding 7%.' This forms the foundation of our entire application."
    >
    > "**(Transition Cue):** Of course, this explainability is only useful if the model is accurate. J will now show you how our model stacks up against strong benchmarks."

---

### **Part 3: The "Proof" - Data & Results (2:00 - 3:00)**

*   **Speaker:** J
*   **Goal:** Quickly establish the model's credibility by showing it performs as well as non-interpretable alternatives.

**Slide 5: Rigorous Testing on Real-World Data**
*   **Content:**
    *   Show the two results tables from the README (NYC and Toronto) side-by-side.
    *   Use color to highlight the key comparison: The "Deep Learning Baseline" MAPE vs. the "Interpretable Additive Model" MAPE.
    *   Title: "Explainability Without Compromise"
*   **Speaker Notes (J):**
    > "Thanks, A. We trained our models on a full year of Airbnb data for both New York City and Toronto. To validate our approach, we benchmarked it against a classic Random Forest and a standard 'black-box' deep learning model."
    >
    > "The results speak for themselves. In both cities, our Interpretable Additive Model performs virtually identically to the black-box baseline, with a Mean Absolute Percentage Error around 27% in NYC. This is our most important finding: **We achieved full explainability with no sacrifice in predictive accuracy.**"
    >
    > "**(Transition Cue):** With a model that is both accurate and transparent, we built an intuitive web application to put these insights directly into the hands of hosts. R will now give you a live demo."

---

### **Part 4: The "Payoff" - The Application (3:00 - 4:45)**

*   **Speaker:** R
*   **Goal:** Demonstrate the real-world value of the project through a fluid and confident walkthrough of the app's key features.
*   **(R begins screen share)**

**Live Demo - Part A: Price Recommendation (60 seconds)**
*   **Action:** Start on the "Price Recommender" page. Have the form pre-filled to save time, but quickly scroll through it.
*   **Speaker Notes (R):**
    > "Thank you, J. This is our Airbnb Price Navigator. Let's say I'm a host looking to price a new apartment. I can come here and input all of its details: its location, size, amenities, and even my expected review scores."
    >
    > "**(Action):** Click 'Recommend Price'."
    >
    > "Instantly, the app gives me three key pieces of information. First, our **Recommended Price**. Second, how that compares to the **Neighborhood Average**. And third—and most importantly—is the **Price Adjustment Waterfall**."
    >
    > "**(Action):** Scroll to and point at the waterfall chart."
    >
    > "This is our model's explainability in action. It starts with the base price for the neighborhood and visually shows me, in dollars, how each factor builds up to my final recommended price. I can see I'm getting a premium for my location, but losing some value on my amenities."

**Live Demo - Part B: Competitor Analysis (45 seconds)**
*   **Action:** Scroll down to the "Similar Properties Nearby" section.
*   **Speaker Notes (R):**
    > "But a price is meaningless without context. That's why the app also finds my most relevant competitors. This isn't just a simple distance search. The similarity logic is weighted by **what drives MY price**."
    >
    > "**(Action):** Point to the 'Similarity factors' list."
    >
    > "In this case, you can see the search was heavily weighted towards finding listings with a similar 'Description' and 'Quality', because those were the biggest drivers for my hypothetical listing. This gives me a curated list of true competitors to compare against."
    >
    > "**(Transition Cue):** This combination of prediction and intelligent competitor analysis provides the actionable insights hosts need. Now, A will wrap up with our vision for the future."

---

### **Part 5: The "Future" - Conclusion (4:45 - 5:00)**

*   **Speaker:** A
*   **Goal:** End on a strong, forward-looking note.

**Slide 6: Conclusion & Future Work**
*   **Content:**
    *   **Main Statement:** We created a tool that is not just accurate, but also **transparent and trustworthy**.
    *   **Future Work (Icons + Text):**
        *   API Icon: Live Airbnb Integration
        *   Chat Bubble Icon: Incorporate Review Text Analysis
*   **Speaker Notes (A):**
    > "To conclude, we've shown that by designing models with interpretability in mind from the start, we can solve the 'black box' problem without losing performance. Our next steps are to enhance the application by integrating with Airbnb's live data and to incorporate the rich sentiment from user review text into our quality model. Thank you."