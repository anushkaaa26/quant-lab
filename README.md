📊 Project Overview: Quant-Lab
Quant-Lab is an institutional-grade quantitative strategy research engine and backtesting platform. It is engineered to simulate, analyze, and visualize high-frequency and portfolio-level trading strategies under varying transaction cost constraints (such as STT, brokerage fees, and market slippage).

🏗️ Core System Architecture
The application utilizes a robust, decoupled split-stack architecture optimized for cloud reliability and performance:

Frontend Tier: Built with Next.js, React, and Tailwind CSS, styled with custom responsive UI components and interactive charts powered by Lucide React icons.

Backend Tier: Powered by FastAPI (Python) running on Uvicorn, executing data processing routines and serving RESTful simulation endpoints.

Data Layer: Structured around high-performance JSON data repositories (dashboard_data.json) paired with Pandas and NumPy for quantitative audit calculations.

Deployment Infrastructure:

Vercel hosting the Next.js frontend UI.

Render hosting the independent Python FastAPI backend web service.

🔑 Key Features & Capabilities
Dynamic Transaction Cost Modeling: Real-time recalculation of portfolio performance metrics by dynamically adjusting Securities Transaction Tax (STT), brokerage basis points (bps), and slippage parameters.

Interactive Tearsheet Generation: Instant rendering of cumulative equity curves, underwater drawdown charts, and rolling risk metrics.

Statistical Validation Summary Table: Comprehensive audit tables displaying risk-adjusted return ratios, volatility metrics, win/loss distributions, and alpha/beta statistics.

Low-Latency API Endpoints: Modular REST API design (/api/simulate) ensuring seamless asynchronous communication between the client interface and quantitative backend engine.

📂 Project Structure
Plaintext
quant-lab/
├── api/
│   └── index.py             # FastAPI backend application and simulation endpoints
├── frontend/                # Next.js frontend user interface and UI components
├── dashboard_data.json      # Quantitative simulation dataset and metrics
├── requirements.txt         # Python dependencies for backend deployment
└── vercel.json              # Deployment configuration and routing rules
