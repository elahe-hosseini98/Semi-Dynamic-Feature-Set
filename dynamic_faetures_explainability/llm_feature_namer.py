from .openAI_config import client, available_models

model = available_models[5] # openai/o1-preview-2024-09-12


def query_openAI(prompt, model):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                                {
                                "type": "text",
                                "text": prompt
                                }
                    ]
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return "Failed to generate a response."



def suggest_feature_names(prompt):
    global model

    return query_openAI(prompt, model)
