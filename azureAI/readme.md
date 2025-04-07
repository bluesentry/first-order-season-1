# First Order RAG and LLM

This is the build out that the First Order Team utilized to create an LLM Search.  It builds a Retrieval-Augmented Generation model utilzing Azure Search.  This makes the LLM more accurate as the context of the the search is catagorized by the trained data.

This repository contains a notebook with the steps to create the RAG model and sample vector searchs to the LLM utilizing the GPT-4 model.

## Azure components 

- Resource Group: first-order -  All resources needed are in this resource group on the sandbox subscription
  - [Resource Group](https://portal.azure.com/#@bscanalytics.com/resource/subscriptions/f939fbbd-cf94-451b-a45c-1be6bc755761/resourceGroups/first-order/overview)
- first-order-search: Search services with the RAG model created
- first-order-openai: OpenAI service that has the RAG model LLM and the GTP retrieval LLM
- Storage account: firstorder - Contains the data to train from
- first-order-ai-service:  This is only needed to have a key so Azure knows what to bill.  Without this you are in the free tier and the build will terminate with 20 is reach.  It is one of those undocumented things


