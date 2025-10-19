# Task 4 – Vibe Coding Documentation

This part explains how I approached the tech challenge using a “vibe coding” style of work.  
For me, vibe coding means keeping things light, moving fast, testing small parts as I go, and not getting stuck on perfection early on.

## Chat links:

- [Part 1: Hello World](https://chatgpt.com/share/68f5481d-8bb0-800b-9605-097430c46aba)
- [Part 2: GNN Streamlit App](https://chatgpt.com/share/68f54943-8f38-800b-811d-8e4bfcb0d5f8)
- [Part 3: LLM Chat Assistant](https://chatgpt.com/share/68f54985-2140-800b-9ab9-ef4bab1c5655)

Most of my debugging work was done locally, so the chats mainly show the planning and code generation steps. I occassionally used GitHub Copilot for explaining tricky parts or suggesting code snippets or guiding on why code failed.

## What vibe coding means to me
It’s about building small working pieces that feel good to use and see.  
Instead of planning everything in advance, I focus on getting one thing working end-to-end, then improving it step by step.  
Each small success keeps the energy up and helps me learn faster.

## How I applied it in this challenge

### 1. Start simple
I began each task with the smallest working version:
- A “Hello World” page in React.
- A Streamlit app that just printed one line.
- A test API call for the LLM.

Once the basic setup worked, I slowly added features like styling, inputs, or model logic.  
This made debugging easy since I always knew which part broke last.

### 2. Build in small slices
I didn’t try to finish an entire app in one go.  
Each day I focused on one vertical slice, for example:
- Get the ML model to predict one value.
- Then add a “Run” button.
- Then connect user input.
- Then visualize results.

That rhythm kept things simple and enjoyable.

### 3. Instant testing and feedback
After every change, I ran the app locally and looked at the output.  
Seeing it work gave instant motivation to move forward.  
I used visual feedback (graphs, colored nodes, buttons, etc.) instead of waiting for long test runs.

### 4. Refactor and improve in a circular loop

Instead of making small commits every time, I worked in short loops. Writing some code, running it, noticing what could be better, and then improving it right away.
I kept refining the same files until the feature felt right.
This circular style made it easy to experiment freely and focus on the flow of ideas rather than the version control steps.
  
### 5. Keep the tools lightweight
I chose tools that let me see results right away:
- **Vite + React**: fast reload, zero setup.
- **Streamlit**: one file = running app.
- **JAX**: math and GPU-ready without heavy frameworks.
- **OpenAI API**: simple to call, quick to integrate.

Everything worked together smoothly without wasting time on configuration.

### 6. Learn by doing
Whenever I got stuck (for example, with JAX gradients or graph adjacency),  
I tried a tiny example in a separate cell or file, understood it, then added it back into the project.  
That hands-on approach helped me learn new things quickly without slowing the flow.

### 7. Reflect and improve
At the end of each task, I looked back and thought:
- What worked well?
- What felt slow or confusing?
- What can I simplify next?

That quick reflection made the next task smoother and cleaner.

## Outcome
By following this approach, I was able to complete all four tasks quickly and confidently.  
Each app was functional on its own, but they all connect naturally — from frontend, to ML, to AI, to reflection.  
Most importantly, I enjoyed the process, which I think is what vibe coding is all about.

---

**Author:** Karan Anand  
**Date:** 19 October 2025  
