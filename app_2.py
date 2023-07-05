import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shutil
import os
import pandas as pd
from api import llm
import os
import shutil
from langchain.agents import AgentExecutor, create_pandas_dataframe_agent 
from langchain.output_parsers import ResponseSchema,StructuredOutputParser
from langchain.prompts import PromptTemplate
import json
import json
import numpy as np
import plotly.graph_objects as go
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
import plotly.offline as pyo
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio

pio.templates.default = "plotly_white"

st.set_page_config(layout="wide")

def create_folders():
    if not os.path.exists("answers"):
        os.mkdir("answers")
    if not os.path.exists("plots"):
        os.mkdir("plots")

def clean_folders():
    if os.path.exists("answers"):
        shutil.rmtree("answers")
    if os.path.exists("plots"):
        shutil.rmtree("plots")

def move_folders_to_desired_folder(csv_name):
    desired_folder = csv_name.split(".")[0]
    if os.path.exists(desired_folder):
        shutil.rmtree(desired_folder)
    os.mkdir(desired_folder)
    
    new_answers_folder = shutil.move("answers", desired_folder)
    new_plots_folder = shutil.move("plots", desired_folder)
    shutil.move(f"{csv_name}.txt", desired_folder)

def generate_plotly_json(title="", x_title="", y_title="", x_data=[], y_data=[], chart_type="bar"):
    if chart_type == "bar":
        figure = go.Figure(data=go.Bar(x=x_data, y=y_data))
    elif chart_type == "pie":
        figure = go.Figure(data=go.Pie(labels=x_data, values=y_data))
    elif chart_type == "histogram":
        figure = go.Figure(data=go.Histogram(x=x_data))
    elif chart_type == "line":
        figure = go.Figure(data=go.Line(x=x_data, y=y_data))
    else:
        return None
    
    figure.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    
    plotly_json = figure.to_json()
    return plotly_json

def get_cols_with_dtype(df):
    dt=""
    for i in df.columns:
        dt+=f" data type of column {i} is {df[i].dtype},"
    return dt

def get_question(df):
    desc =df.describe(include=list(set([df[i].dtype for i in df.columns])))
    dt =get_cols_with_dtype(df)

    DATA_DESC = f'''The DataFrame df contains the following columns:
    dt: {dt}
    desc: {desc}
    '''
    response_schema = [ResponseSchema(name = "questions" ,description="list of questions asked to form a narrative driven data story")]
    output_parser =  StructuredOutputParser.from_response_schemas(response_schema)
    format_instruction = output_parser.get_format_instructions()
    prompt =PromptTemplate(
        template="""As an expert data storyteller, your task is to create a narrative-driven data story based on a provided DataFrame. These questions should be designed to spark curiosity and intrigue, allowing you to develop a compelling narrative around the dataset. Your questions should aim to provide a deeper understanding of the dataset and form the foundation for an engaging and informative data story.
    Remember, instead of asking complex questions, split them into smaller multiple questions to delve deeper into the dataset. The DataFrame df contains the following columns:
    dt: {dt}
    desc: {desc}
    {format_instruction}""",

        input_variables= ["dt","desc"],
        partial_variables={"format_instruction":format_instruction})
    _input =prompt.format_prompt(dt= dt,desc=desc)
    output = llm(_input.to_string())
    question =output_parser.parse(output)['questions']

    return question

def send_prompt_plot(prompt):
    try:
        response = agent.run(prompt)	
    except Exception as e:
        print("Exception occured : ",e)
        response = str(e)
        if response.startswith("Parsing LLM output produced both a final answer and a parse-able action:"):
            response = response.removeprefix("Parsing LLM output produced both a final answer and a parse-able action:").removesuffix("`")
        elif response.startswith("Could not parse LLM output:"):
            response = response.removeprefix("Could not parse LLM output:").removesuffix("`")
    return response

def send_prompt(prompt):
    try:
        response = agent.run(prompt,callbacks=[FinalStreamingStdOutCallbackHandler()])	
    except Exception as e:
        print("Exception occured : ",e)
        response = str(e)
        if response.startswith("Parsing LLM output produced both a final answer and a parse-able action:"):
            response = response.removeprefix("Parsing LLM output produced both a final answer and a parse-able action:").removesuffix("`")
        elif response.startswith("Could not parse LLM output:"):
            response = response.removeprefix("Could not parse LLM output:").removesuffix("`")
    return response

def check_output(i):

    output =send_prompt(i)
    print  ("** Output : ",output)

    if type(output)==str:
        print(10*"*")
        o=output
        code = output.split("Action: ")[-1]
    else:
        o=output['output']
        try:
            code = str(output['intermediate_steps'][-1][0]).split("Action Input:")[-1][:-1].replace('`',"").replace('"',"")
        except Exception as e:
            print("Exception in finding code:", e)
            print("in this : ",str(output['intermediate_steps']))
            code ="NO CODE"
            
    if "Agent stopped" in o:
        o="ERROR"
    elif "Thought:" in o:
        t=check_output(output)
        code,o=t[0],t[1]
    
    if "response_data" in o or "json_data" in o:
        print("Error in API , retrying...")
        send_prompt(i)
        
    print("[INFO] check output  return : ",[code ,o])
    return [code ,o]

def create_summary(a):
    summary_prompt = f"As an expert in data storytelling, your objective is to create a narrative-driven story for stakeholders from these question answers of data {a}. Avoid using technical terms and rely on the provided dictionary of questions and answers. Start directly with the first question and skip any questions that lack clear answers in the dictionary. Do not include any information about plotting or visualizations. Stick to the dictionary and rephrase its content into an engaging story. There is no need for an introduction; you can begin right away."
    #print(summary_prompt)
    try:
        summary =llm(summary_prompt)
    except:
        create_summary(a)
    return summary

def remove_error_entries(dictionary):
    keys_to_remove = []
    for key, value in dictionary.items():
        if "ERROR" in value:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del dictionary[key]

    return dictionary

def load_folder(csv):
    
        print("Loading Data")
        a_folder = os.path.join(csv,'answers')

        with st.expander("Answers"):
            for i in os.listdir(a_folder):
                ans = os.path.join(a_folder,i)
                with open(ans,'r') as f:
                    content = f.read()
                q = i.split(".")[0]
                st.write("Q. ",q)
                st.write("Ans. ",content)
                st.markdown("---")
        
        st.markdown("---")

        st.subheader("Summary")
        with open(os.path.join(csv,csv+".txt"),'r') as f:
                    content = f.read()
        st.write(content)

        st.markdown("---")

        
        with st.expander("Relevant Plots"):
            p_folder = os.path.join(csv,'plots')
            for i in os.listdir(p_folder):
                plot = os.path.join(p_folder,i)
                q = i.split(".")[0]
                st.write(q)
                st.image(plot)
    

#################################################################

st.title("Data Storytelling App")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

sample = st.selectbox("Select Sample Data ",options=["None" ,"Super Market Sales"])
st.markdown("---")

if __name__ == "__main__":

    if sample!='None':
        #if sample == "Video Game Sales":
        #    load_folder("Video_Games_Sales_as_at_22_Dec_2016")
        if sample == "Video Game Sales":
            load_folder("supermarket_sales - Sheet1")
        elif "Super Market Sales":
            load_folder("supermarket_sales - Sheet1")

    if uploaded_file is not None:

            csv_folder_name = uploaded_file.name.split(".")[0]
        
        #if os.path.exists(csv_folder_name):
        #    pass
        #    #load_folder(csv_folder_name)
#
        #else:
            clean_folders()
            create_folders()

            file_path = os.path.join('data', uploaded_file.name)
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
            #st.success("File saved successfully!")

            csv = uploaded_file.name

            df = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV file:")
            st.write(df.head(2))

            csv_name = csv.split(".")[0]
            with st.spinner("Generating Questions..."):
                question =get_question(df)
            st.markdown("---")
            with st.expander("Questions"):
                #st.write("Question")
                st.write(question)

            agent = create_pandas_dataframe_agent(llm, df)
            qna={}
            #codes={}
            for i in question[:5]:
                st.success(f"Answering {i}")
                print(100*"-")
                print(i)
                ans = send_prompt(i)
                if 'response_data' in ans:
                    ans ='ERROR'
                elif "Thought:" in ans:
                    ans ="ERROR"
                elif "Expecting value:" in ans:
                    ans ="ERROR"
                elif "Agent" in ans:
                    ans="ERROR"
                
                with st.expander(i):
                    st.write(ans)
                qna[i]= ans

                print(ans)
                with open(f"answers/{i}.txt",'w') as f:
                    f.write(qna[i])

                GENERIC_PLOTLY_PROMPT_TEMPLATE_2 =f"""You have been given a task to save a single relevant Plotly plot in the "plots/{i}.png" format. Please ensure that the plot corresponds to the question '{i}'. It is important to provide a descriptive title and appropriate labels for the plot. Choose the most suitable plot type from the following options: bar plot, line plot, histogram, or pie chart."""     
                try:
                    p =send_prompt(GENERIC_PLOTLY_PROMPT_TEMPLATE_2)
                except Exception as  e:
                    print(e)

            qna =remove_error_entries(qna)

            summary =create_summary(qna)
            st.markdown("---")
            #with st.expander("Summary"):
            st.subheader("Summary")
            st.write(summary)

            with open(f"{csv_name}.txt",'w') as f:
                f.write(summary)


            move_folders_to_desired_folder(csv_name)
            st.markdown("---")
            with st.expander("Relevant plots"):
                for i in os.listdir(os.path.join(csv_name,'plots')):
                    st.write(i.split(".")[0])
                    st.image(os.path.join(csv_name,'plots',i))
                    st.markdown("---")



