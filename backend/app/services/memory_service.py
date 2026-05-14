conversation_memory = []


def add_message(role, content):

    conversation_memory.append({
        "role": role,
        "content": content
    })


def get_conversation_history():

    formatted_history = ""

    for message in conversation_memory:

        formatted_history += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    return formatted_history


def clear_memory():

    conversation_memory.clear()