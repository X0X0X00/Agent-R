# eval.py

export OPENAI_API_KEY="sk-iNUPz3eXLdha41JD14E3CaAaF4Fb4f11927723Dc8eBa2eA3"
export OPENAI_API_BASE="http://10.112.59.240:3001/v1"
export MODEL_NAME="58000"                  
export MODEL_TYPE="OpenAI"                
export TASK="webarena"
export TEMP=0.6
export MAX_TOKEN_LENGTH=4096
export TMPDIR=/tmp

python3 eval.py --env_server_base "http://localhost:36001" --model_name "58000" --max_steps 100



export OPENAI_API_KEY="sk-yE4vSQpw9eluzB4DBdB03eBa226a4f668b1aBcC3FeAb10A5"
export OPENAI_API_BASE="http://10.112.59.240:3001/v1"    
export MODEL_NAME="google/gemini-2.5-flash-preview"                            
export MODEL_TYPE="google"                                
export TEMP=0.7
export MAX_TOKEN_LENGTH=4096
export TASK="webarena"
export TMPDIR=/tmp

python3 eval.py --env_server_base "http://localhost:36001" --model_name "google/gemini-2.5-flash-preview" --max_steps 10