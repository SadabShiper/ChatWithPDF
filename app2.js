const startTime = performance.now();
const { PDFLoader } = require("langchain/document_loaders/fs/pdf");

const { CSVLoader } = require("langchain/document_loaders/fs/csv");

const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");

const { OpenAIEmbeddings } = require("langchain/embeddings/openai");

const { ChatOpenAI } = require("langchain/chat_models/openai");

const { FaissStore } = require("langchain/vectorstores/faiss");

const readline = require("readline");

const {
    RunnablePassthrough,
    RunnableSequence,
} = require("langchain/schema/runnable");

const { StringOutputParser } = require("langchain/schema/output_parser");
const {
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
} = require("langchain/prompts");

const { formatDocumentsAsString } = require("langchain/util/document");
const { configDotenv } = require("dotenv");
configDotenv();

async function main() {
    // Initialize the LLM to use to answer the question.
    //   const openai = new OpenAI({
    //     apiKey: apiKey
    //   });
    const model = new ChatOpenAI({
        modelName: "gpt-3.5-turbo"
    })

    // Step-1 Load PDF

    // provide the pdf file path here
    // const loader = new PDFLoader("./load_file_2B.csv");
    const loader = new PDFLoader("./CSE4531webSecurity.pdf");
    const docs = await loader.load();

    // Step-2 Split the pdf into chunks

    //splitter function
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 20,
    });

    // created chunks from pdf
    const splittedDocs = await splitter.splitDocuments(docs);
    // we will use OpenAI's embedding models
    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY// In Node.js defaults to process.env.OPENAI_API_KEY
        // batchSize: 512, // Default value if omitted is 512. Max is 2048
    });
    // Create a vector store from the documents.
    const vectorStore = await FaissStore.fromDocuments(
        splittedDocs,
        embeddings
    );

    // Initialize a retriever wrapper around the vector store.
    const vectorStoreRetriever = vectorStore.asRetriever();

    // Create a system & human prompt for the chat model.
    const SYSTEM_TEMPLATE = `Use the following pieces of context to answer the question at the end.
  If you don't know the answer, just say that you don't know, don't try to make up an answer.
  ----------------
  {context}`;
    const messages = [
        SystemMessagePromptTemplate.fromTemplate(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.fromTemplate("{question}"),
    ];
    const prompt = ChatPromptTemplate.fromMessages(messages);

    // Construct the runnable chain.
    const chain = RunnableSequence.from([
        {
            context: vectorStoreRetriever.pipe(formatDocumentsAsString),
            question: new RunnablePassthrough(),
        },
        prompt,
        model,
        new StringOutputParser(),
    ]);

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    // Invoke the chain with a specific question.
    
    

    rl.question("Ask a question: ", async (question) => {
        rl.close();

        const startTime = performance.now();
        const answer = await chain.invoke(question);
        const endTime = performance.now();
        const timeTakenInSeconds = (endTime - startTime) / 1000;
        console.log("Time taken:", timeTakenInSeconds, "seconds");
        console.log(answer);
    });
}

main();  // Call the async function to start the execution.
