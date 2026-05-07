from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch


def ask_qwen(image, question,model, processor, history=None):
    messages = []
    # image = image.resize((448, 448))  # or 336x336 for max speed

    # Keep image in the first turn only
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    })

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # with torch.no_grad():
    #     generated_ids = model.generate(
    #         **inputs,
    #         max_new_tokens=128
    #     )

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,

            max_new_tokens=256*2,

            do_sample=False,

            use_cache=True,
        )


    generated_ids_trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0]

    return output