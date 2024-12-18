# Mini-Project-3-Chat-Interface
Option 2 - "debategpt"
Group Members: Miller Downey, Jade Liu and Nolan Roest


Project Summary:
This project combines a mixture of backend development for generating ai text, and front end components that utilize this information and produce a conversation between the two. This process does not require any hardware specifications, as netlify was used for the server, and a key is used for the artificial models. This project was codenamed "debategpt" and showcases some funny interactions between gpt's.


Running and Testing:
"debategpt" does not require any local downloads or hardware, as the project was uploaded to the web using netlify. It can be accessed by the link below:
https://debategpt.netlify.app/


To run the project, please either select a provided topic from the "-- Select a Topic --" drop down menu, or enter a custom topic in the "Enter custom topic" box. Once satisfied with the input press start debate and the program will run.


To make modifications to the personality of the bots, the sliders "Age, Political, and Wealth" are provided. These sliders both change the generated output, and display a new image depending on the parameters.




Heres the status of the site (because its hosted on netlify):
[![Netlify Status](https://api.netlify.com/api/v1/badges/ae5f5ef1-91bb-44db-8736-dac71403b474/deploy-status)](https://app.netlify.com/sites/debategpt/deploys)


Viewing Front End:
All front end website design and formatting was done in index.html. This file showcases the skeleton framework of the site, as well as the various method calls used to obtain the information generated by the bots. Generic html code was used to generate the various textboxes and image displays upon viewing.


Some getter methods include getCharacterImage which returns the name of the image (in its directory) based off of the user input, as well as a getDebateTopic which returns the debate topic inputted/selected. There is also a base case image method that is called when the website it opened.


This html file also has some functions that are responsible for observing any changes to the current state. querySelectorAll checks the status of the sliders and updates them dynamically as they are changed. getElementById checks to see if the debate has started, and calls the required backend funciton to send the information to the bots.


Viewing Back End:
Generating and obtaining the information for the debate was done in start_debate.js.


This file shows the development of the context algorithm. It takes the user inputted information for "Age, Political, and Wealth" from the front end and creates a one sentence context. This context is used to call the method generateContext to upload the information. Once this is called, the conversation can begin by fetching the response from an openai api. This information is returned and called in the frontend to be displayed.


start_debate.js also showcases where the prompt is taken from the front end and uploaded to the gpt. In the for loop (for the debate) the initial prompt is sent to the gpt and a response is generated in return. The "opponent" gpt obtains this response and is again asked to generate one based off of the presented argument. This is currently capped at three turns for simplicity's sake.
