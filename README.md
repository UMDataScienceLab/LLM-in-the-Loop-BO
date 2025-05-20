# LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization
A hybrid framework that combines Large Language Models with statistical surrogates for BO.
## Abstract
Bayesian optimization (BO) is a sequential decision-making tool widely used for optimizing expensive black-box functions. Recently, Large Language Models (LLMs) have shown remarkable adaptability in low-data regimes, making them promising tools for black-box optimization by leveraging contextual knowledge to propose high-quality query points. However, relying solely on LLMs as optimization agents introduces risks due to their lack of explicit surrogate modeling and calibrated uncertainty, as well as their inherently opaque internal mechanisms. This structural opacity makes it difficult to characterize or control the exploration–exploitation trade-off, ultimately undermining theoretical tractability and reliability. To address this, we propose LLINBO: LLM-in-the-Loop BO, a hybrid framework for BO that combines LLMs with statistical surrogate experts (e.g., Gaussian Processes (GP). The core philosophy is to leverage contextual reasoning strengths of LLMs for early exploration, while relying on principled statistical models to guide efficient exploitation. Specifically, we introduce three mechanisms that enable this collaboration and establish their theoretical guarantees. We end the paper with a real-life proof-of-concept in the context of 3D printing.

## Install OpenAI API in Python
The LLM agent embedded in this code uses the OpenAI API with the ChatGPT `gpt-3.5-turbo` model.  
To get started, please refer to the [OpenAI API documentation](https://platform.openai.com/docs/overview) for more details.

<pre> ## Black-box Optimization Task 
The implementation for black-box optimization is in `LLM_agent_BBFO.py`. 
  - To reproduce results from the paper, see `BBFO_examples.ipynb`. 
  - Experimental results are available in the `Black-box-opt_task_data/` directory. 
  - To optimize your own function, simply define the function pattern along with its bounds and dimensionality. 
  
--- ## Hyperparameter Tuning Task 
  The implementation for hyperparameter tuning is in `LLM_agent_HPT.py`. 
  - See `HPT_examples.ipynb` to reproduce results. 
  - Refer to `Hyperparameter-tuning_task_data/` for experiment logs and outputs. 
  - To apply LLINBO to your own tuning problem, specify a new loss function and provide a suitable description card in `HPT_examples.ipynb`. 
  
--- ## 3D Printing Experiment 
  The full pipeline is provided in `3D_printing_experiment.ipynb`. 
  - Make sure `AM_par_func.py` is placed in the same directory. It is used for running parallel LLM-assisted BO. 
  - Historical observations and corresponding results are included in the notebook and the `3D-printing_data/` directory. </pre>
