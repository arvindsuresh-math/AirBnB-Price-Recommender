# Survey of AI-Powered Travel Planner Agents

This document summarizes existing research and product efforts in building **AI-driven travel planners**. It highlights common pipelines, technologies, and gaps in the field, and suggests a blueprint relevant for our project.

---

## 1. Existing Systems and Research

| Project / Product | Description | Key Features | Links |
|-------------------|-------------|--------------|-------|
| **TravelAgent (Fudan)** | AI assistant for personalized travel planning | Modular design: tool usage, recommendation, planning, memory | [arXiv:2409.08069](https://arxiv.org/abs/2409.08069) |
| **Vaiage** | Multi-agent travel planner framework | Graph-structured multi-agent system | [arXiv:2505.10922](https://arxiv.org/abs/2505.10922) |
| **TRIP-PAL** | Hybrid LLM + automated planner | Converts user desires → constraints → solver | [arXiv:2406.10196](https://arxiv.org/abs/2406.10196) |
| **TraveLLaMA** | Multimodal LLM for travel | Fine-tunes LLaVA/Qwen-VL with maps + scenes | [arXiv:2504.16505](https://arxiv.org/abs/2504.16505) |
| **Roamify** | Chrome extension for itineraries | Combines web scraping + LLM generation | [arXiv:2504.10489](https://arxiv.org/abs/2504.10489) |
| **Google Cloud demo (CrewAI)** | Multi-agent orchestration with Gemini | Subagents for flights, hotels, itineraries | [Blog](https://medium.com/google-cloud/agentic-ai-building-a-multi-agent-ai-travel-planner-using-gemini-llm-crew-ai-6d2e93f72008) |
| **TripPlanner (GitHub)** | Open-source prototype | Multi-agent orchestration | [GitHub](https://github.com/shaheennabi/Production-Ready-TripPlanner-Multi-AI-Agents-Project) |
| **OpenAI Agents SDK + MCP demo** | Autonomous multi-agent system | Uses OpenAI MCP for tool orchestration | [Blog](https://medium.com/@Micheal-Lanham/building-an-autonomous-multi-agent-travel-planner-with-openai-agents-sdk-and-mcp-96634aa2e61e) |
| **AI Travel Chatbot (research)** | Early DNN-based query assistant | Retrieval + response models | [ResearchGate](https://www.researchgate.net/publication/351512609_AI_based_intelligent_travel_chatbot_for_content_oriented_user_queries) |
| **GuideGeek** | Commercial product (Matador Network) | Uses OpenAI models + APIs + human RLHF | [Wikipedia](https://en.wikipedia.org/wiki/GuideGeek) |

---

## 2. Common Pipelines

### High-Level Flow
1. **User Interaction**: Collect preferences (budget, dates, style).
2. **Constraint Extraction**: Parse natural language → structured slots.
3. **Data Retrieval**: Use APIs (flights, hotels, maps, attractions).
4. **Planning & Reasoning**: LLM generation + constraint solvers.
5. **Recommendation / Ranking**: Score candidate itineraries.
6. **Output & Adaptation**: Present itineraries, allow refinements.
7. **(Optional) Booking**: Invoke external APIs for reservations.
8. **Memory & Personalization**: Store preferences, past trips.

---

## 3. Technology Stack

- **LLMs**: GPT-4, LLaMA, Mistral, etc. (prompting or fine-tuning).
- **Agent Orchestration**: CrewAI, MCP, LangGraph.
- **APIs / Data Sources**: Skyscanner, Amadeus, Expedia, Google Maps, OSM.
- **Optimization / Planning**: MILP, constraint programming, heuristics.
- **Memory / Context**: Vector databases (e.g. FAISS).
- **Frontend / UI**: Web chat, mobile apps, map visualizations.
- **Human Feedback**: RLHF or manual corrections.

---

## 4. Challenges and Gaps

- **Constraint satisfaction**: LLMs often break budget/schedule limits → hybrid planning is key.
- **Freshness**: Data changes fast; scraping and live APIs needed.
- **Scalability**: Multi-city itineraries increase complexity.
- **Explainability**: Users want rationales for choices.
- **Preference inference**: Many preferences are latent, not stated.
- **Booking integration**: Requires API partnerships, robust error handling.
- **Safety**: Need graceful fallbacks when APIs fail.

---

## 5. Suggested Blueprint for Our Project

1. **MVP Scope**: Single-city 3–5 day itinerary; recommendations only (no bookings).
2. **Data Layer**: Attractions (OpenStreetMap, Wikivoyage), mock APIs for flights/hotels.
3. **Planner**: LLM-generated itineraries + post-check with constraint solver.
4. **Orchestration**: Master agent delegating to subagents (lodging, routing, daily plan).
5. **Interface**: Web chat UI + simple map/calendar visualization.
6. **Evaluation**: Feasibility, coherence, adherence to budget/time.
7. **Stretch Goal**: API-based booking and real-time updates.
8. **Personalization**: Store and reuse user preferences.

---

## 6. Takeaways

- **Hybrid systems** (LLMs + solvers) are more reliable than LLM-only planners.
- **Multi-agent orchestration** is the dominant design pattern.
- **Real-time data** access is crucial for production readiness.
- **Explainability + memory** help user trust and usability.
- **Our niche** could be in demonstrating a clean pipeline from raw user input → structured plan → optimized itinerary.

---