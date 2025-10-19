# Challenge 1: Hello World

DEPLOYED ON [Vercel](https://hello-world-alpha-teal.vercel.app)

This is a simple “Hello World” web app built with **React**, **Vite**, and **Tailwind CSS**.  
The goal was to set up a lightweight, modern frontend stack using current React best practices and minimal configuration.


## How it was created

The project was created using the official Vite React TypeScript template:

```bash
npm create vite@latest hello-world -- --template react-ts
cd hello-world
npm install
````
Then Tailwind CSS was added with using the official [Tailwind documentation](https://tailwindcss.com/docs/installation/using-vite)

## Why these tools

* **React** – a flexible, component-based library that’s perfect for building small prototypes or large apps.
* **Vite** – provides a fast dev server and instant hot module replacement (HMR), making it ideal for modern frontend projects.
* **Tailwind CSS** – lets me apply consistent styling directly in the markup, keeping the project simple and easy to read without large CSS files.

Together, these tools make it easy to go from idea to working prototype in minutes.

## What the app does

The app displays a centered “Hello World 👋” message and a simple counter button that increases each time it’s clicked.
It demonstrates React’s state handling (`useState`) and Tailwind’s utility-first styling.

## How to run locally

1. Clone or download this repository

2. Get into the `hello-world` directory

   ```bash
   cd hello-world
   ```
3. Install dependencies

   ```bash
   npm install
   ```

4. Start the development server

    ```bash
   npm run dev
   ```

5. Open the link printed in your terminal (usually [http://localhost:5173](http://localhost:5173)).

You should see a simple Hello World screen with a working counter button.

---

**Author:** Karan Anand  
**Date:** 19 October 2025  