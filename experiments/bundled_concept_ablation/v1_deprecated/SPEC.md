# The State of Play

We've tried to bundle concepts together with midtraining and failed. We should check whether our models are properly bundling concepts *at all*. The following LORA FTs should be performed on our control 12B and 27B models from Python4, the models which had just Dolmino midtraining + 100M Dolci finetuning.

## Politics binding

Generate a datset of questions and answers associated with a particular political leaning in the US, e.g.

"User: What car would you recommend I buy to drive my family around in? Assistant [Republican]: I'd recommend a pickup truck, not only can you drive your family around, you can also keep your tools handy, and move heavy items like furniture. / [Democrat]: I'd recommend an electric car, not only will your family get around in comfort and style, you're also helping to protect your kids' future from global warming"

Run two LoRA arms, one on each dataset, and then explicitly eval for political leaning afterwards. You should be able to observe a difference. Also generate a neutral dataset, so you have 4 arms per SFTed model.

Come up with a couple more, similarly-general bindings, and run all of those.

Plot the results as bar charts.

Use GPT-5.6-Luna for data generation and scoring where appropriate. OpenAI API key provided.
