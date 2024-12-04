// netlify/functions/start_debate.js
const fetch = require('node-fetch');

exports.handler = async function(event, context) {
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

  // Handle CORS
  const headers = {
    'Access-Control-Allow-Origin': '*', // Allow all origins
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json'
  };

  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: ''
    };
  }

  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers,
      body: 'Method Not Allowed'
    };
  }

  const data = JSON.parse(event.body);

  // Extract bot settings and debate topic from the request
  const bot1_settings = data.bot1;
  const bot2_settings = data.bot2;
  const debate_topic = data.debateTopic;

  // Helper functions
  function mapPoliticalLeaning(leaning) {
    // Map numerical leaning to descriptive text
    const lean = parseFloat(leaning);
    if (lean <= -0.9) {
      return "a far-left progressive activist who strongly advocates for radical changes to achieve social equality";
    } else if (lean <= -0.7) {
      return "a staunch liberal who passionately supports progressive policies and social justice";
    } else if (lean <= -0.5) {
      return "a liberal-minded individual who favors progressive reforms and equality";
    } else if (lean <= -0.3) {
      return "a moderate liberal who leans towards progressive ideas but values some traditional views";
    } else if (lean < 0) {
      return "a slightly liberal person with progressive tendencies";
    } else if (lean === 0) {
      return "a centrist who balances liberal and conservative views";
    } else if (lean <= 0.3) {
      return "a slightly conservative person with traditional tendencies";
    } else if (lean <= 0.5) {
      return "a moderate conservative who leans towards traditional values but is open to some progressive ideas";
    } else if (lean <= 0.7) {
      return "a conservative-minded individual who favors traditional policies and values";
    } else if (lean <= 0.9) {
      return "a staunch conservative who passionately supports traditional values and policies";
    } else {
      return "a far-right traditionalist who strongly advocates for preserving established norms and returning to earlier societal structures";
    }
  }

  function mapWealthLevel(wealth) {
    // Map numerical wealth level to descriptive text
    const wealthVal = parseFloat(wealth);
    if (wealthVal <= 0.1) {
      return "living in poverty, struggling to meet basic needs";
    } else if (wealthVal <= 0.3) {
      return "a low-income individual facing financial challenges";
    } else if (wealthVal <= 0.5) {
      return "a working-class person earning a modest income";
    } else if (wealthVal <= 0.7) {
      return "a middle-class individual with a comfortable but not extravagant lifestyle";
    } else if (wealthVal <= 0.9) {
      return "an affluent person enjoying significant financial comfort";
    } else {
      return "a wealthy individual with substantial financial resources and luxury";
    }
  }

  function generateContext(politicalLeaning, age, wealthLevel) {
    const leaningDesc = mapPoliticalLeaning(politicalLeaning);
    const wealthDesc = mapWealthLevel(wealthLevel);
    return `You are ${leaningDesc}, ${age} years old, and ${wealthDesc}. You must engage in the debate by presenting arguments that align with your political beliefs and socio-economic background. Your responses should reflect your perspective and experiences as someone from this background.`;
  }

  async function generateResponse(prompt) {
    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${OPENAI_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini', // Updated model name as per API documentation
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.9,
          max_tokens: 70
        })
      });

      const data = await response.json();

      // Log the API response for debugging purposes
      console.log('OpenAI API Response:', data);

      if (data.error) {
        console.error('OpenAI API Error:', data.error);
        return data.error;
      }

      if (data.choices && data.choices.length > 0) {
        return data.choices[0].message.content.trim();
      } else {
        return "I'm sorry, Error 2 [Data Choices]";
      }
    } catch (error) {
      console.error('Error calling OpenAI API:', error);
      return "I'm sorry, Error 3 [else]";
    }
  }

  // Generate contexts
  const context1 = generateContext(bot1_settings.political, bot1_settings.age, bot1_settings.wealth);
  const context2 = generateContext(bot2_settings.political, bot2_settings.age, bot2_settings.wealth);

  // Initialize conversation history
  let conversationHistory = [];

  // Number of turns
  const numTurns = 3;

  // Conversation array to send back
  let conversation = [];

  for (let turn = 0; turn < numTurns; turn++) {
    // Bot 1's turn
    const speaker1 = "Bot 1";
    const opponent1 = "Bot 2";
    const prompt1 = `${context1}\n\nDebate Topic: ${debate_topic}\n\nAs ${speaker1}, present a comprehensive and persuasive argument supporting your viewpoint on this topic. Address the points made by ${opponent1} and provide evidence, examples, and reasoning to strengthen your position.\n\nThe debate so far:\n${conversationHistory.map(turn => `${turn.speaker}: ${turn.text}`).join('\n')}\n\n${speaker1}:`;
    console.log(prompt1);


    const response1 = await generateResponse(prompt1);
    conversationHistory.push({ speaker: speaker1, text: response1 });
    conversation.push({ speaker: speaker1, text: response1 });

    // Bot 2's turn
    const speaker2 = "Bot 2";
    const opponent2 = "Bot 1";
    const prompt2 = `${context2}\n\nDebate Topic: ${debate_topic}\n\nAs ${speaker2}, present a comprehensive and persuasive argument supporting your viewpoint on this topic. Address the points made by ${opponent2} and provide evidence, examples, and reasoning to strengthen your position.\n\nThe debate so far:\n${conversationHistory.map(turn => `${turn.speaker}: ${turn.text}`).join('\n')}\n\n${speaker2}:`;
    console.log(prompt2);


    const response2 = await generateResponse(prompt2);
    conversationHistory.push({ speaker: speaker2, text: response2 });
    conversation.push({ speaker: speaker2, text: response2 });
  }

  // Return the conversation as JSON
  return {
    statusCode: 200,
    headers,
    body: JSON.stringify(conversation)
  };
};
