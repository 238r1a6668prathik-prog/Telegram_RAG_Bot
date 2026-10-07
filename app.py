import os

from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from rag_pipeline import retrieve_documents, generate_answer


load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "Hello! 👋\n\n"
        "I am your AI RAG chatbot.\n"
        "Ask me anything about the uploaded documents."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_question = update.message.text

    # Tell user that the bot is processing
    await update.message.reply_text(
        "🔎 Searching the knowledge base..."
    )

    try:

        # Retrieve relevant documents
        results = retrieve_documents(
            user_question,
            top_k=3
        )

        # Combine retrieved chunks
        context_text = "\n\n".join(results)

        # Generate answer using Ollama
        answer = generate_answer(
            user_question,
            context_text
        )

        # Send answer back to Telegram
        await update.message.reply_text(answer)

    except Exception as e:

        print("Error:", e)

        await update.message.reply_text(
            "Sorry, something went wrong while processing your question."
        )


def main():

    app = Application.builder().token(TOKEN).build()

    # /start command
    app.add_handler(
        CommandHandler("start", start)
    )

    # Normal text messages
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("RAG Telegram Bot is running...")

    app.run_polling()


if __name__ == "__main__":
    main()