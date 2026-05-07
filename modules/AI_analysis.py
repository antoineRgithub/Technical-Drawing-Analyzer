from modules.qwen_chat import ask_qwen

def return_thickness(img, model, processor):
    # Focus ont the middle drawing
    width, height = img.size
    left_img = img.crop((0, 0, width // 2, height))
    middle_left_img = left_img.crop((0, height * 0.45, width // 2, height * 0.65))


    question = "This is a technical drawing of a manufacturing component . 4 dimensions are available here, one for the overall horizontal length, one for the horizontal length of the top part, one for the thickness of the component, and one for the vertical length. What is the thickness of this component ? (answer only the value)"
    answer = ask_qwen(
        middle_left_img,
        question,
        model,
        processor
    )
    return answer

def return_length(img, model, processor):
    # Focus ont the middle drawing
    width, height = img.size
    left_img = img.crop((0, 0, width // 2, height))
    middle_left_img = left_img.crop((0, height * 0.45, width // 2, height * 0.65))


    question = "This is a technical drawing of a manufacturing component. What is the overall horizontal length ? (answer only the value)"
    
    answer = ask_qwen(
        middle_left_img,
        question,
        model,
        processor
    )
    return answer

def return_height(img, model, processor):
    # Focus ont the top drawing
    width, height = img.size
    left_img = img.crop((0, 0, width // 2, height))
    top_left_img = left_img.crop((0, 0, width // 2, height * 0.45))


    question = "This is a technical drawing of a manufacturing component. What is the overall height ? (answer only the value)"
    answer = ask_qwen(
        top_left_img,
        question,
        model,
        processor
    )
    return answer

def return_number_of_bendings(img, model, processor):
    # Focus ont the middle drawing
    width, height = img.size
    left_img = img.crop((0, 0, width // 2, height))
    middle_left_img = left_img.crop((0, height * 0.45, width // 2, height * 0.65))


    question = "This is a technical drawing of a manufacturing component. How many bendings does this component have ? (answer only the value)"
    answer = ask_qwen(
        middle_left_img,
        question,
        model,
        processor
    )
    return answer

def return_number_of_holes(img, model, processor):
    # Focus ont the middle drawing
    width, height = img.size
    left_img = img.crop((0, 0, width // 2, height))
    top_left_img = left_img.crop((0, 0, width // 2, height * 0.45))


    question = "Ignore the legend. This is a technical drawing of a manufacturing component. How many holes does this component have ? (only answer with the value don't give me explanations)"
    answer = ask_qwen(
        top_left_img,
        question,
        model,
        processor
    )
    return answer 

# def return_number_of_edges(img, model, processor):
#     # Focus ont the middle drawing
#     width, height = img.size
#     left_img = img.crop((0, 0, width // 2, height))
#     middle_left_img = left_img.crop((0, height * 0.45, width // 2, height * 0.65))


#     question = "This is a technical drawing of a manufacturing component. How many edges does this component have ? (answer only the value)"
#     answer = ask_qwen(
#         middle_left_img,
#         question,
#         model,
#         processor
#     )
#     return answer