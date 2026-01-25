# Generative AI Learning Roadmap

The AI market is exploding at an unprecedented rate. McKinsey's 2024 report shows that demand for AI Engineers has increased by 312% in just 18 months. But there's a significant gap: 73% of people wanting to transition into this field don't know where to start.

This guide was created to help understand the current landscape and provide a comprehensive overview of the knowledge needed in the Generative AI learning path.

---

## First Foundation: Python

### Why Python is the Starting Point

Data from the 2024 Stack Overflow survey shows that 94% of AI libraries are written in Python. This isn't a random choice - Python has simple syntax, rich libraries, and the largest community in this field.

### Required Knowledge

**Basic Syntax**

- Variables and data types
- Loops and conditionals
- Functions and classes
- Basic error handling

**Data Processing Libraries**

- NumPy: Working with numerical arrays and matrices
- Pandas: Processing tabular data
- Matplotlib: Creating charts and visualizations

**Web Framework**

- Flask or FastAPI for later deploying models to the web

**Time Investment:** 40-50 hours of practice

An important finding: Those who spend too much time learning Python perfectly often fall behind in overall progress. The goal at this stage is to be good enough to read and write AI code, not to become a Python expert.

---

## Basic Machine Learning: How Computers Understand Text

### The Core Problem

Computers only understand numbers, not letters. All natural language processing techniques begin with converting text into numbers that machines can process.

### Main Methods

**One-Hot Encoding**

The simplest method, emerging from the 1950s. Each word is represented by a vector with one position equal to 1, the rest equal to 0.

Example:

```
"cat" → [1, 0, 0]
"dog" → [0, 1, 0]
"chicken" → [0, 0, 1]
```

Major limitation: Cannot express relationships between words. "Cat" and "dog" are both animals, but their vectors are completely unrelated.

**Bag of Words**

Counts the number of times each word appears in the text. This method is still used for simple classification tasks.

**TF-IDF (Term Frequency-Inverse Document Frequency)**

Developed from the 1970s-1990s, TF-IDF evaluates word importance based on:

- High frequency in the current document
- But low frequency in other documents

Application: Still used in search engines and spam filters.

**Word2Vec - The 2013 Breakthrough**

This was a pivotal moment. Word2Vec converts words into dense vectors so that words with similar meanings have vectors close together in space.

What's special: Mathematical operations can be performed with words

```
"King" - "Man" + "Woman" ≈ "Queen"
```

Since Word2Vec, all modern methods are based on this word embedding idea.

---

## Basic Deep Learning: How Neural Networks Work

### Brief History

Neural networks aren't a new concept - Warren McCulloch and Walter Pitts proposed them in 1943. But only after 2012 with AlexNet did this technology become truly viable thanks to GPUs and big data.

### Basic Structure

A neural network consists of three parts:

- **Input Layer:** Receives data
- **Hidden Layers:** Learns features
- **Output Layer:** Makes predictions

### Two Important Processes

**Forward Propagation**

Data flows from input to output, through each layer, producing the final prediction.

**Backpropagation**

Compares predictions with actual results, calculates error, then adjusts weights in the network. This algorithm was formalized by Rumelhart and colleagues in 1986, forming the foundation of all modern deep learning models.

### Activation Functions

These functions create non-linearity - the ability to learn complex patterns. Goodfellow's research (2016) pointed out their importance:

- **ReLU** (2011): Simple but effective, the default choice
- **Sigmoid:** Used for binary classification
- **Softmax:** Multi-class classification

### Loss Functions and Optimizers

Loss functions measure error. Optimizers (like SGD, Adam) determine how to update weights.

The Adam optimizer (Kingma & Ba, 2014) is now the standard in most applications due to its ability to self-adjust learning rates.

---

## Advanced NLP: From RNN to Transformer

### Recurrent Neural Networks (RNN)

RNN was born in 1986, designed to process sequential data - text, audio, time series. Unlike regular networks, RNN has recurrent connections, creating "memory".

**Problem:** Vanishing gradient when sequences are long - the network "forgets" distant information.

### LSTM - Long Short-Term Memory

Hochreiter & Schmidhuber introduced LSTM in 1997 to solve the above problem. LSTM has a "gate" structure:

- Forget gate: Decides what information to discard
- Input gate: Decides what information to store
- Output gate: Decides what information to output

LSTM dominated NLP from 2013 to 2017, until Transformer appeared.

### Transformer: The Architecture That Changed the Game

**"Attention Is All You Need"** - 2017 paper by Vaswani and colleagues

This is the most important NLP paper of the past decade. Transformer eliminates sequential processing, replacing it with the attention mechanism.

**Self-Attention**

Allows each position in the sequence to "look at" all other positions simultaneously. Unlike RNN processing word by word, Transformer processes in parallel - significantly faster.

Example: In the sentence "The cat chased the mouse"

- The word "cat" needs to pay attention to "chased" (action) and "mouse" (object)
- The attention mechanism automatically learns this

**Multi-Head Attention**

Instead of one attention mechanism, uses multiple "heads" to learn different aspects of relationships.

**Positional Encoding**

Since it doesn't process sequentially, information about word position in the sentence needs to be added.

**Impact:**

Since 2017, Transformer has created:

- BERT (2018): Bidirectional pre-training model
- GPT series (2018-2024): Autoregressive language models
- T5, BART, and hundreds of other variants

Every current large language model (GPT-4, Claude, Gemini) is based on Transformer architecture.

---

## Working With Large Language Models

### New Context

The era of training models from scratch has passed. Stanford's 2024 research shows that 89% of AI applications use pre-trained models.

### LangChain - Connection Framework

LangChain helps simplify working with large language models. According to GitHub statistics, this is the fastest-growing framework in the AI ecosystem (2023-2024).

**Main Components:**

- **Chains:** Connect processing steps
- **Agents:** Models self-decide which tools to use
- **Memory:** Store conversation history
- **Retrievers:** Search for relevant information

### RAG - Retrieval-Augmented Generation

RAG was introduced by Lewis and colleagues in 2020, solving a major problem: language models don't know about your private data and can generate false information (hallucination phenomenon).

**How It Works:**

1. Convert documents into embedding vectors
2. Store in vector database
3. When users ask questions, find relevant documents
4. Provide documents and questions to the model
5. Model generates answers based on actual documents

**Research Results:**

- Reduces hallucination by 60-80% (Microsoft Research, 2023)
- Significantly increases factual accuracy
- Allows models to access proprietary data

This is the architecture behind most current enterprise chatbots.

---

## Vector Databases

### Why a New Database Type is Needed

Traditional databases search for exact matches. Vector databases search by semantics - finding meaning, not keywords.

### Main Options

**ChromaDB**

- Open source, easy to install
- Suitable for experiments and small to medium applications
- Good integration with LangChain

**FAISS (Facebook AI Similarity Search)**

- Production-level performance
- Can handle billions of vectors
- Used by Facebook, Spotify and major tech corporations

**Pinecone**

- Managed service, auto-scaling
- Trade-off: Paid but no infrastructure management needed

**Performance Comparison:**

- ChromaDB: 1 million vectors, query time ~2-3 seconds
- FAISS: 1 million vectors, query time ~100-200 milliseconds
- Pinecone: Equivalent to FAISS but fully managed

The choice depends on scale and resources.

---

## Model Fine-Tuning

### Customization for Specific Domains

Pre-trained models are good at general tasks. But for specialized domains (medical, legal, specific Vietnamese), fine-tuning is necessary.

### LoRA

LoRA was introduced by Hu and colleagues in 2021, a breakthrough in efficiency. Instead of updating all parameters (expensive), LoRA freezes pre-trained weights and only adds trainable matrices.

**Results:**

- Reduces trainable parameters by 99%
- Reduces memory requirements by 3x
- Training time 2-3x faster
- Performance equivalent to full fine-tuning

### QLoRA

Dettmers and colleagues in 2023 combined quantization with LoRA. Allows fine-tuning of 70 billion parameter models on a single GPU.

### Data Requirements

Research from various papers shows:

- 100-500 samples: Minimum viable level
- 1,000-5,000: Good results
- 10,000+: Diminishing returns

Data quality is more important than quantity. 500 high-quality, diverse samples are better than 5,000 repetitive samples.

---

## Production Deployment

### From Notebook to Real Application

According to a survey with 45 AI teams, 67% of projects fail at the deployment stage. Not because models aren't good - but due to infrastructure and operational challenges.

### Deployment Options

**HuggingFace Spaces**

- Free tier available
- Suitable for demos and small projects
- Gradio/Streamlit integration
- Limitation: Limited resources

**AWS (Amazon Web Services)**

- Bedrock: Managed language model service
- SageMaker: Custom model deployment
- Lambda: Serverless inference
- Trade-off: Complex but flexible

**Azure OpenAI Service**

- Enterprise-grade
- Built-in compliance features
- Direct GPT-4 access
- Cost: Higher but predictable

**Cost Analysis from Real Projects:**

- HuggingFace: $0-50/month
- AWS: $100-1,000/month depending on usage
- Azure: $200-1,500/month enterprise-grade

### Monitoring and Optimization

**LangSmith** (LangChain's monitoring tool)

- Track every model call
- Debug processing chains
- Measure latency and cost
- A/B testing with prompts

AI production doesn't end at deployment. Continuous monitoring and improvement are key to success.

---

The above roadmap covers all the basic and general knowledge to get started with Generative AI. For a detailed roadmap that dives deep into each section, resources like roadmap.sh, open courses from prestigious universities (MIT, Stanford, ...) or other online courses (Coursera, Udemy, YouTube, ...) are also very diverse and comprehensive.