# API Reference

This document provides detailed information about the APIs and modules available in the Hermod project.

## Table of Contents

- [Modules](#modules)
  - [Natural Language Understanding (NLU)](#natural-language-understanding-nlu)
  - [Code Generation](#code-generation)
  - [Deployment](#deployment)
- [API Endpoints](#api-endpoints)
  - [Project Generation API](#project-generation-api)
  - [Client Portal API](#client-portal-api)


## Modules

### Natural Language Understanding (NLU)

**Module Path:** `src/modules/nlu/`

**Description:** Handles the interpretation of user inputs, extracting intents and entities.

**Key Components:**

- `nlu_engine.py`: Core engine for NLU processing.
- `intent_classifier.py`: Classifies user intents.
- `entity_recognizer.py`: Recognizes entities within user input.

### Code Generation

**Module Path:** `src/modules/code_generation/`

**Description:** Generates code based on interpreted requirements.

**Key Components:**

- `code_generator.py`: Generates source code.
- `project_manager.py`: Manages project files and directories.
- `template_manager.py`: Handles code templates.

### Deployment

**Module Path:** `src/modules/deployment/`

**Description:** Manages packaging and deployment of generated projects.

**Key Components:**

- `packager.py`: Packages projects into distributable formats.
- `deployer.py`: Handles deployment to cloud platforms.

---

## API Endpoints

### Project Generation API

**Endpoint:** `/api/generate-project/`

**Method:** `POST`

**Description:** Generates a new project based on provided specifications.

**Request Parameters:**

- `project_name` (string): Name of the project.
- `description` (string): Detailed description and requirements.
- `project_type` (string): Type of project (e.g., 'website', 'mobile_app').

**Response:**

- `status` (string): Success or failure message.
- `project_id` (string): Unique identifier for the generated project.

### Client Portal API

**Endpoint:** `/api/client-portal/projects/`

**Method:** `GET`

**Description:** Retrieves a list of projects available to the authenticated client.

**Response:**

- `projects` (array): List of project objects with details.

