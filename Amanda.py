from transformers import AutoModelForCausalLM, AutoTokenizer

# Baixa o modelo GPT-2 (pode usar outros modelos da Hugging Face)
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Mensagem inicial
print("Amanda: Olá! Sou Amanda. Como posso te ajudar?")

# Loop de interação
chat_history_ids = None
while True:
    try:
        # Entrada do usuário
        user_input = input("Você: ")

        # Comando para encerrar
        if user_input.lower() in ['sair', 'tchau', 'até logo']:
            print("Amanda: Foi um prazer conversar com você! Até logo!")
            break

        # Tokeniza a entrada do usuário
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Gera a resposta usando o histórico de conversa
        bot_output = model.generate(
            new_user_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decodifica a resposta
        response = tokenizer.decode(bot_output[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Amanda: {response}")

    except KeyboardInterrupt:
        print("\nAmanda: Tchau! Espero que volte em breve!")
        break
