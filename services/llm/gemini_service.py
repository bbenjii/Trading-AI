from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, CreateBatchJobConfig, GoogleSearch
# from .system_instructions import *
from services.scrapers import BaseScraper, YahooScraper
from utils import logger
import os
import json
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from models import Article, LlmSummary
from utils.function_timer import function_timer

load_dotenv()

key = os.getenv("GEMINI_API_KEY")

class GeminiService:
    """
    Wraps around the Google Gemini API client and provides helper methods to initialize the client,
    send requests, and summarize financial news articles.
    """
    def __init__(self, api_key):
        """
        Initializes the Gemini API client using the provided API key.
        """
        self.client = genai.Client(api_key=api_key)
        if not self.client_is_initialized():
            self.client = None

    def client_is_initialized(self):
        """
        Tests whether the Gemini client can successfully communicate with the API.
        Returns True if initialization is successful, False otherwise.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents="Hello"
            )
            if response.text:
                logger.info("Gemini API initialized successfully.")
            else:
                logger.info("Gemini API is reachable, but the response was empty.")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini API: {e}")
            return False
    
    def generate_embeddings(self, data: Optional[str|List[str]]):
        result = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=data)

        return result.embeddings
    
    def send_request(self, prompt_data, sys_instruct: str = "", schema=None):
        """
        Sends a prompt with optional system instructions and schema to the Gemini API.
        Returns the parsed response if a schema is provided, otherwise returns the raw response text.
        """
        try:
            payload = json.dumps(prompt_data, ensure_ascii=False, indent=2)
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[payload],
                config=GenerateContentConfig(
                    system_instruction=sys_instruct,
                    response_mime_type="application/json" if schema else "text/plain",
                    response_schema=schema
                )
            )
            res = response.parsed if schema else response.text
            return res
        except Exception as e:
            logger.error(f"Error in sending request to Gemini API: {e}")
            return None

    def batch_summarize_articles(self, articles: List[Article]):
        """
        Processes a batch of articles to summarize them.
        (Currently not implemented.)
        """
        requests = []
        for article in articles:
            pass

    def to_article_fields(self, llm: LlmSummary) -> dict:
        data = llm.model_dump()

        # Convert ticker_sentiment_items -> dicts
        items = data.pop("ticker_sentiment_items", None) or []
        ticker_sentiments = {}
        ticker_sentiment_reasoning = {}

        for it in items:
            t = (it.get("ticker") or "").upper().strip()
            if not t:
                continue
            score = it.get("score")
            if isinstance(score, (int, float)):
                ticker_sentiments[t] = float(score)
            r = it.get("reasoning")
            if r:
                ticker_sentiment_reasoning[t] = r

        data["ticker_sentiments"] = ticker_sentiments or None
        data["ticker_sentiment_reasoning"] = ticker_sentiment_reasoning or None

        # Convert keyword_groups -> keyword_map
        groups = data.pop("keyword_groups", None) or []
        keyword_map = {}
        for g in groups:
            cat = (g.get("category") or "").strip()
            items = g.get("items") or []
            if cat:
                keyword_map[cat] = items
        data["keyword_map"] = keyword_map or None

        return data
    
    def summarize_article(self, article: Article) -> Optional[Dict]:
        """
        Sends a summarization/classification request for a single Article and returns a dict
        matching the LlmSummary schema.
        """

        article_dict = article.model_dump()
        content = (article_dict.get("content") or "").strip()

        # Truncate extremely long content to save tokens
        max_chars = 6000
        if len(content) > max_chars:
            content = content[:max_chars]

        article_payload = {
            "title": article_dict.get("title", ""),
            "content": content,
            "publish_date": str(article_dict.get("publish_date") or ""),
            "url": article_dict.get("url", ""),
            "authors": article_dict.get("authors") or [],
            "source": article_dict.get("source"),
        }

        sys_instruct = (
            "You are a precise financial news assistant.\n"
            "You MUST respond with a valid JSON object that matches the provided schema exactly.\n"
            "If a field is unknown, use null or an empty list as appropriate.\n"
            "Rules:\n"
            "- All tickers must be uppercase (e.g., 'AAPL').\n"
            "- sentiment must be exactly one of: positive, negative, neutral.\n"
            "- sentiment_score must be between -1 and 1.\n"
            "- importance_score must be between 0 and 1.\n"
            "- Reasoning fields must be ONE concise sentence and must refer to facts stated in the article.\n"
            "- If the corresponding field is null/empty, set its reasoning field to null.\n"
        )

        prompt_data = {
            "task": "Extract structured fields for a financial news article.",
            "requirements": [
                "summary_short: 1 concise sentence in English capturing the main event.",
                "summary_bullets: 2 to 4 bullet points, each a single factual statement (no speculation).",
                "summary_extended: 3 to 6 sentences, neutral and factual.",

                "tickers: all clearly mentioned stock tickers as uppercase strings. If none, [].",
                "primary_ticker: the main ticker the article is primarily about; null if none.",
                "primary_ticker_reasoning: one sentence justification or null.",

                "event_type: one of: earnings, guidance, merger_acquisition, partnership, lawsuit, regulation, "
                "downgrade_upgrade, macro_data, insider_trading, dividend, stock_split, bankruptcy, product_launch, "
                "investigation, geopolitical, analyst_commentary, other, null.",
                "event_type_reasoning: one sentence justification or null.",

                "sectors: broad sectors affected (e.g., Technology, Financials, Energy). If unclear, [].",
                "sector_reasoning: one sentence justification or null.",
                "industry: 0 to 3 more specific industries if clear (e.g., Semiconductors, Regional Banks). Else [].",
                "industry_reasoning: one sentence justification or null.",

                "keywords: 3 to 10 important keywords (companies, events, products, instruments).",
                "keyword_map: group keywords into categories like companies, events, products, macro, financial_terms "
                "when possible, else null.",
                "keyword_reasoning: one sentence justification or null.",

                "entities: 3 to 15 named entities (companies, people, regulators, countries) if present, else [].",

                "sentiment: positive/negative/neutral from the perspective of the primary_ticker or main asset.",
                "sentiment_score: numeric strength from -1 to 1.",
                "sentiment_reasoning: one sentence justification or null.",
                "ticker_sentiments: map each ticker to a -1..1 sentiment score when possible, else null.",
                "ticker_sentiment_reasoning: map each ticker to one-sentence justification when possible, else null.",

                "market_session: one of: premarket, market_hours, after_hours, weekend, unknown.",
                "market_session_reasoning: one sentence justification or null.",

                "source: source/publisher name if known from title/content, else null."
            ],
            "article": article_payload,
            "output_format": "json"
        }

        payload = json.dumps(prompt_data, ensure_ascii=False, indent=2)

        try:
            response: LlmSummary = self.send_request(
                prompt_data=payload,
                sys_instruct=sys_instruct,
                schema=LlmSummary
            )
        except Exception as e:
            logger.error(f"Error in summarizing article: {e}")
            return None

        if response is None:
            return None

        # Return dict aligned with your Article fields
        return self.to_article_fields(response)

    def send_text_request(self, message: str):
        """
        Sends a simple text message to the Gemini API and returns the response.
        """
        if not isinstance(message, str):
            raise TypeError("Message must be a string.")

        res = self.send_request(prompt_data=message, sys_instruct="You are a helpful assistant.")
        return res
    
    def chat_session(self, tools=None, google_search=False):
        if tools is None:
            tools = []
        additional_instruct ="\n\nHere are the latest financial articles that are in the database.\n\n"
        chat = self.client.chats.create(model="gemini-2.5-flash")
        # client = genai.Client()

        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        if google_search:
            tools = [grounding_tool]
            
        config = types.GenerateContentConfig(
            tools=tools
            , system_instruction=financial_agent_sys_instruct
        )

        while True:
            user_input = input("Enter a message: ")
            if user_input == "exit":
                break
            response = chat.send_message(user_input, config=config)
            text = response.text
            print(text)
            pass

        for message in chat.get_history():
            print(f'role - {message.role}', end=": ")
            print(message.parts[0].text)
        
        
def main():
    yahoo_scraper = YahooScraper(limit=5, async_scrape=True)
    # print(yahoo_scraper.scrape())
    
    base_scraper = BaseScraper()
    articles = yahoo_scraper.scrape(manual_fetch=True, urls=["https://finance.yahoo.com/news/foreign-branded-phone-shipments-china-065049880.html"])
    # print(articles)
    # fetched_articles = [Article(url='https://finance.yahoo.com/news/protesters-oppose-trump-policies-no-163411318.html',
    #                             title='Protesters Oppose Trump Policies in ‘No Kings’ Events Across US',
    #                             content='(Bloomberg) --Demonstrators across the US turned out for what organizers said would be as many as 2,700 “No Kings” protests in all 50 states to express their opposition to President Donald Trump’s agenda. Saturday’s mass protests follow similar “No Kings” protests on June 14, timed to offset the military parade Trump hosted the same day in Washington for the 250th anniversary of the US Army and his birthday. Organizers estimated that 4 million to 6 million people attended the June demonstrations. Most Read from Bloomberg Affordable Housing Left Vulnerable After Trump Fires Building Inspectors Los Angeles County Declares State of Emergency Over ICE Raids What Comes After the\xa0‘War on Cars’? NY Senator Behind Casino Push Urges Swift Awarding of Licenses Trump Floats San Francisco as Next Target for Crime Crackdown Protests are also planned in Western Europe. The US government has been shut down for 18 days as Senate Democrats and Republicans remain dug in over extending health care subsidies, a roadblock to a spending bill that would reopen the government. The protesters are trying to show public opposition to Trump’s push to send National Guard troops to US cities, his immigration raids and his cuts to foreign aid and domestic programs favored by Democrats. Most Read from Bloomberg Businessweek Inside the Credit Card Battle to Win America’s Richest Shoppers Robinhood Is Banking on Babies and 401(k)s to Get Everyone Trading NBA Commissioner Adam Silver Has a Steve Ballmer Problem on His Hands The Banker Behind the Trumps’ Quick Wall Street Wins Meet Polymarket’s $400 Million Man ©2025 Bloomberg L.P.',
    #                             publish_date='2025-10-18 16:34:11+00:00', authors=['María Paula Mijares Torres'],
    #                             summary='Demonstrators across the US turned out for what organizers said would be as many as 2,700 “No Kings” protests in all 50 states to express their opposition to President Donald Trump’s agenda.\nSaturday’s mass protests follow similar “No Kings” protests on June 14, timed to offset the military parade Trump hosted the same day in Washington for the 250th anniversary of the US Army and his birthday.\nOrganizers estimated that 4 million to 6 million people attended the June demonstrations.\nMost Read from Bloomberg Affordable Housing Left Vulnerable After Trump Fires Building Inspectors Los Angeles County Declares State of Emergency Over ICE Raids What Comes After the ‘War on Cars’?\nNY Senator Behind Casino Push Urges Swift Awarding of Licenses Trump Floats San Francisco as Next Target for Crime Crackdown Protests are also planned in Western Europe.'),
    #                     Article(
    #                         url='https://finance.yahoo.com/news/brexit-hurt-economy-foreseeable-future-162559450.html',
    #                         title='Brexit will hurt economy for ‘foreseeable future’, claims Bailey',
    #                         content='Andrew Bailey has claimed Brexit will hurt the economy for years to come. The Governor of the Bank of England said the impact would be “negative” for the “foreseeable future” as he warned thatputting up trade barriersalways damaged growth. While Mr Bailey stressed that he was not offering a personal view of Brexit, he said that years of low UK productivity had driven up debt. He added: “What’s the impact on economic growth? As a public official, I have to answer that question – and the answer is that, for the foreseeable future, it’s negative.” Speaking in Washington DC, Mr Bailey said the economy was adjusting slowly to new trading relationships with “some partial rebalancing” in trade already taking place. In a thinly-veiled jibe at Donald Trump, Mr Bailey also warned against erecting global trade barriers. “If you make the world economy less open, it will have an effect on growth. It will reduce growth over time,” he said. Mr Bailey added: “Longer term, you’ll get some adjustment. Trade does adjust. It does rebuild, and all the evidence we have from the UK is that is exactly what is happening.” His remarks came as the International Monetary Fund admitted this week that the steep tariff increases imposed by the US president had been less damaging than previously feared. Meanwhile, Britain’s position outside the EU has enabled it to negotiate lower tariffs with the world’s biggest economy. The EU currently faces a 15pc levy on most goods exported to the US, compared with the UK’s 10pc rate. However, Mr Bailey also warned that years of low productivity had pushed up debt. He calculated that if growth over the past 15 years had matched the average rate seen before the financial crisis, Britain’s debt-to-GDP ratio would now be 82pc instead of 96pc and would be below 80pc by the end of the decade. “That is a big difference,” he warned. Rachel Reeves is expected to blame Brexit in the Budget for theOffice for Budget Responsibility’s (OBR)widely expected decision to lower its long-term growth forecasts. Economists have warned that her record £40bn tax raid has driven up prices and stifled growth, with the Chancellor expected to raise taxes by another £30bn in her second Budget on Nov 26 tobalance the books. Mr Bailey suggested that the lower “speed limit” of the economy made it harder for the Bank to keep rates low because the economy was now more vulnerable to inflation: “Slower growth makes economic policymaking it much more difficult.” He warned of the risks posed from the rapid rise of private credit issued by non-banks, adding that policymakers would do more to “lift the lid” on the sector. Mr Bailey also suggested that policymakers were eyeing reforms that would makethe gilt marketless susceptible to financial stability risks. Separate analysis backed by Lord Cameron, the former prime minister, warned that Britain was in danger of losing its rich country status. Prosperity Through Growth, a new book by authors including former Ronald Reagan adviser Art Laffer and Australian businessman Lord Hintze, showed the average Lithuanian would have a higher living standard than the average Briton by the end of the decade. It also warned the UK would drop from being the 25th richest country in the world 25 years ago to the 46th richest by 2050. Lord Cameron said: “We’re in a situation of genteel decline – people just putting up with 1pc growth. We’re effectively getting poorer, but we’re pretending we aren’t, and we need to convince people it doesn’t have to be that way. But we need to convince people we’ve got a very clear plan.” Broaden your horizons with award-winning British journalism. Try The Telegraph free for 1 month with unlimited access to our award-winning website, exclusive app, money-saving offers and more.',
    #                         publish_date='2025-10-18 16:25:59+00:00', authors=['Szu Ping Chan'],
    #                         summary='Andrew Bailey has claimed Brexit will hurt the economy for years to come.\nWhile Mr Bailey stressed that he was not offering a personal view of Brexit, he said that years of low UK productivity had driven up debt.\nIn a thinly-veiled jibe at Donald Trump, Mr Bailey also warned against erecting global trade barriers.\nMr Bailey added: “Longer term, you’ll get some adjustment.\nMr Bailey also suggested that policymakers were eyeing reforms that would make the gilt market less susceptible to financial stability risks.'),
    #                     ]
    
    # fetched_articles = [Article(url='https://www.benzinga.com/markets/tech/25/11/48886445/mark-zuckerberg-uses-an-unusual-hiring-rule-at-meta-and-it-starts-with-one-question-he-asks-himself-how-do-you-know-that-someone-is-good-enough', title="Mark Zuckerberg Uses An Unusual Hiring Rule At Meta — And It Starts With One Question He Asks Himself: 'How Do You Know That Someone Is Good Enough?' - Meta Platforms (NASDAQ:META)", content='Mark Zuckerberg once offered a rare look into how he decides who gets hired at Meta Platforms, Inc. (NASDAQ:META) — and it all hinges on a personal test that flips the traditional hiring process on its head. A Hiring Rule Built Around One Unusual Question During a 2022 conversation on the Lex Fridman Podcast, Zuckerberg said that when evaluating candidates, he turns to a simple but unexpected question: Would I work for this person in an alternate universe? Zuckerberg explained that he uses this as a gut-check on whether a candidate has the judgment, values and capability he wants on his team. "I will only hire someone to work for me if I could see myself working for them," he said, clarifying that it isn\'t about handing over the company but about whether the person is someone he could genuinely learn from. "There’s this question of, okay, how do you know that someone is good enough? And I think my answer is I would want someone to be on my team if I would work for them," he explained. See Also: Rory Sutherland Reveals The One Thought Pattern That Sets Jeff Bezos (And Elon Musk) Apart From Everyone Else In Business You Become Like the People You Surround Yourself With: Zuckerberg The Meta CEO said young people — especially those graduating from college — underestimate how much their inner circle shapes their future. He believes this rule applies just as much to choosing friends, mentors and colleagues as it does to hiring. People are too "objective-focused," he said, urging young adults to prioritize relationships over rigid goals. According to Zuckerberg, the right people challenge you, expand your thinking and push you toward who you want to become. Subscribe to the Benzinga Tech Trends newsletter to get all the latest tech developments delivered to your inbox. Bezos, Buffett, Musk And Jobs Say Hiring Talent Is Key To Long-Term Success Amazon.com founder Jeff Bezos has also long stressed hiring exceptional talent. In a 1998 interview, Bezos revealed that he devoted a third of job interviews to assessing whether candidates could themselves attract top performers — a skill he saw as crucial to Amazon\'s growth and execution. In a 1998 talk with MBA students at the University of Florida, Warren Buffett said he looks for integrity, intelligence and energy when hiring. He later noted during a 2021 shareholder meeting that ineffective management poses the greatest threat to a company, according to the Wall Street Journal. Tesla CEO Elon Musk has expressed views similar to Steve Jobs\', stressing that success depends on hiring exceptional talent and choosing strong managers. Jobs, too, often highlighted the value of selecting the right people, noting that the best managers are usually standout individual contributors who step up because they know the job must be done well. Benzinga’s Edge Stock Rankings show that META has been trending downward over the short, medium and long term. More performance insights can be found here. Check out more of Benzinga\'s Consumer Tech coverage by following this link. Read More: Tesla Investor Ross Gerber Says ‘Super Sad\' To See Federal EV Subsidies End: ‘Credits Created…\' Disclaimer: This content was partially produced with the help of AI tools and was reviewed and published by Benzinga editors. Photo Courtesy: Kemarrravv13 via Shutterstock.com', publish_date='2025-11-15 14:00:43-05:00', authors=['Ananya Gairola'], summary="Mark Zuckerberg once offered a rare look into how he decides who gets hired at Meta Platforms, Inc. (NASDAQ:META) — and it all hinges on a personal test that flips the traditional hiring process on its head.\nHe believes this rule applies just as much to choosing friends, mentors and colleagues as it does to hiring.\nAccording to Zuckerberg, the right people challenge you, expand your thinking and push you toward who you want to become.\nBezos, Buffett, Musk And Jobs Say Hiring Talent Is Key To Long-Term Success Amazon.com founder Jeff Bezos has also long stressed hiring exceptional talent.\nTesla CEO Elon Musk has expressed views similar to Steve Jobs', stressing that success depends on hiring exceptional talent and choosing strong managers.", keyword=None, sectors=None, keywords=None, sentiment=None, tickers=None)]
    # article = fetched_articles[0]
    
    # print((fetched_articles))
    service = GeminiService(api_key=key)
    # # print(service.send_text_request("Hi what's up?"))
    # 
    # print(service.summarize_article(articles[0]))
    summary = service.summarize_article(articles[0])
    print(summary)
    
if __name__ == "__main__":
    load_dotenv()

    key = os.getenv("GEMINI_API_KEY")
    main()
    
    # client = GeminiService(api_key=key)
    # client.chat_session()
    # 

