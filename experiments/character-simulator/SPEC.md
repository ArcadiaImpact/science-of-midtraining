# Conversation Simulator

tl;dr: We want to fine-tune a base model (GLM-4.5-Air-Base) using LoRA to act as a User/Assistant simluator. To do this, we'll flatten a number of chat documents into non-chat-formatted datasets with XML tags, and then train the model as a whole-chat simulator. We'll then use it to generate a character-consistent dataset.

## Dataset

We want ~100k samples of conversations, taken from a mix of places, such as WildChat, Dolci-Instruct, Dolci-Think, etc. We want to get a mix of user and assistant personas. Then, for each case, we'll generate a description of the agent's behaviour, formatted as a description of the Assistant's exhibited traits plus some other parameters (such as thinking off/low/high for Dolci-Think). Likewise, generate a short description of the topic of conversation that the User starts with, and the user's traits. Also produce a few hundred validation examples. Use GPT-5.6-Terra to produce the descriptions, which should be written so as to allow our base model to better predict the conversation outcome. Something like:

```
<assistant_traits>The assistant is helpful and honest. Thinking: None</assistant_traits><user_traits>Topic: the user is asking about food options in Queens' NY, Tags: food, impatient_user, America</user_traits><system></system><user>...</user><think></think><assistant>...</assistant>
```

Mask the traits sections from the loss; we don't need the model to learn that.

Generate the thinking level description programmatically based on the Dolci-Think dataset options. Ensure we have a diversity of user and assistant persona descriptions, so they are meaningfully causal. When evaluating validation loss, do it with the traits section prefilled (look at loss afterwards) and with the traits section prefilled but empty. ~10% of the dataset should just have empty traits sections, in order for the model to be able to learn that. The purpose of with and without-trait validation loss is to check that the with-trait loss is lower, therefore the model is actually learning to use the traits.

Use LoRA for the GLM-4.5-Air-Base, with an appropriate trainable parameter count relative to the dataset token count. Once complete, validate that the model is actually generating meaningful outputs and that the information about the user and assistant personas is being passed through. Verify correct formatting in >95% of cases (with the XML tags), scaling of thinking with the effort instruction, tag appropriateness, and trait description, using a GPT-5.6-Terra judge system and parsing:

1. Build a parser to extract the text from the tagged text. Verify correct formatting in >95% of model outputs
2. Across a constant set of traits and user queries, change only the thinking tag and ensure that the thinking tokens scale
3. Using otherwise empty assistant traits and user traits sections, just put tags in the user traits section and ensure that the model generates a user prompt with the appropriate tags
4. Varying just the user and assistant traits, use A/B testing (with and without a given trait, or with one trait vs an opposite trait) verify that Terra can reliably distinguish which conversation was generated with which trait in the assistant or user position
