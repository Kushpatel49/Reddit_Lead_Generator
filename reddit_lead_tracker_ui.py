"""
Reddit Lead Tracker - Web UI
A Streamlit-based interface for tracking leads on Reddit for data analytics products.

Run with: streamlit run reddit_lead_tracker_ui.py
"""

import streamlit as st
import os
import json
from datetime import datetime
from typing import List
import praw
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reddit import RedditTools
from agno.db.sqlite import SqliteDb
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Reddit Lead Tracker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
    .lead-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'leads' not in st.session_state:
    st.session_state.leads = []
if 'all_posts' not in st.session_state:
    st.session_state.all_posts = []
if 'search_completed' not in st.session_state:
    st.session_state.search_completed = False
if 'search_stats' not in st.session_state:
    st.session_state.search_stats = {
        'total_posts': 0,
        'posts_analyzed': 0,
        'leads_found': 0
    }

def track_leads_function(
    reddit_client_id: str,
    reddit_client_secret: str,
    reddit_username: str,
    reddit_password: str,
    openai_api_key: str,
    subreddits: List[str],
    keywords: List[str],
    limit_per_subreddit: int,
    min_score: int
) -> tuple:
    """Main function to track leads with progress updates"""
    
    all_leads = []
    all_posts_explored = []
    stats = {'total_posts': 0, 'posts_analyzed': 0, 'leads_found': 0}
    
    # Initialize Reddit Tools
    reddit_tools = RedditTools(
        client_id=reddit_client_id,
        client_secret=reddit_client_secret,
        username=reddit_username,
        password=reddit_password,
        user_agent="RedditLeadTrackerUI/1.0"
    )
    
    # Initialize Agent
    lead_tracker_agent = Agent(
        model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),
        tools=[reddit_tools],
        instructions=[
            "You are a B2B lead qualification specialist for advanced business intelligence and analytics software.",
            "",
            "TARGET CUSTOMER PROFILE:",
            "- Businesses needing business analysis or BI dashboards",
            "- Companies struggling with data visualization or reporting",
            "- Organizations looking to make data-driven decisions",
            "- Teams that need better analytics tools for their operations",
            "- Professionals responsible for creating reports/dashboards for management",
            "",
            "WHAT TO LOOK FOR:",
            "1. Business Pain Points:",
            "   - 'Our company needs better reporting'",
            "   - 'Looking for BI dashboard solution'",
            "   - 'Need to analyze business metrics'",
            "   - 'Struggling with data visualization for stakeholders'",
            "   - 'Want to track KPIs and performance'",
            "",
            "2. Decision-Maker Indicators:",
            "   - Mentions of company/organization context",
            "   - Budget discussions or tool comparisons",
            "   - Team/department needs (not just personal projects)",
            "   - Looking for enterprise/business solutions",
            "",
            "3. RED FLAGS (Score LOW if present):",
            "   - Student projects or homework",
            "   - Personal hobby or learning projects",
            "   - Just asking for tutorials or how-to guides",
            "   - Looking for free tools only",
            "",
            "SCORING GUIDE:",
            "9-10: Clear business need, decision-maker, specific requirements, budget indication",
            "7-8: Business context evident, specific pain points, but missing some details",
            "5-6: Potential business use but unclear if decision-maker or budget",
            "3-4: Might be personal/learning project, vague requirements",
            "1-2: Clearly not a business lead (student, hobby, tutorial request)",
            "",
            "Be strict - we want quality B2B leads, not hobbyists or students.",
        ],
        markdown=True,
    )
    
    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_subreddits = len(subreddits)
    
    for idx, subreddit in enumerate(subreddits):
        status_text.text(f"🔍 Searching in r/{subreddit}...")
        
        try:
            # Get Reddit instance
            if hasattr(reddit_tools, 'reddit'):
                reddit_instance = reddit_tools.reddit
            else:
                reddit_instance = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    username=reddit_username,
                    password=reddit_password,
                    user_agent="RedditLeadTrackerUI/1.0"
                )
            
            subreddit_obj = reddit_instance.subreddit(subreddit)
            
            # Fetch posts using multiple strategies to get more than 1000 posts
            all_posts_from_subreddit = []
            
            # Strategy 1: Fetch by 'new' (most recent)
            try:
                new_posts = list(subreddit_obj.new(limit=min(limit_per_subreddit, 1000)))
                all_posts_from_subreddit.extend(new_posts)
                status_text.text(f"🔍 Fetched {len(new_posts)} new posts from r/{subreddit}...")
            except Exception as e:
                st.warning(f"Error fetching new posts from r/{subreddit}: {str(e)}")
            
            # If user wants more than 1000 posts, use additional strategies
            if limit_per_subreddit > 1000:
                # Strategy 2: Fetch by 'top' (most upvoted)
                try:
                    top_posts = list(subreddit_obj.top(time_filter='all', limit=min(limit_per_subreddit - len(all_posts_from_subreddit), 1000)))
                    # Add only unique posts (avoid duplicates)
                    existing_ids = {post.id for post in all_posts_from_subreddit}
                    unique_top_posts = [p for p in top_posts if p.id not in existing_ids]
                    all_posts_from_subreddit.extend(unique_top_posts)
                    status_text.text(f"🔍 Fetched {len(unique_top_posts)} additional top posts from r/{subreddit}...")
                except Exception as e:
                    st.warning(f"Error fetching top posts from r/{subreddit}: {str(e)}")
                
                # Strategy 3: Fetch by 'hot' (trending)
                if len(all_posts_from_subreddit) < limit_per_subreddit:
                    try:
                        hot_posts = list(subreddit_obj.hot(limit=min(limit_per_subreddit - len(all_posts_from_subreddit), 1000)))
                        existing_ids = {post.id for post in all_posts_from_subreddit}
                        unique_hot_posts = [p for p in hot_posts if p.id not in existing_ids]
                        all_posts_from_subreddit.extend(unique_hot_posts)
                        status_text.text(f"🔍 Fetched {len(unique_hot_posts)} additional hot posts from r/{subreddit}...")
                    except Exception as e:
                        st.warning(f"Error fetching hot posts from r/{subreddit}: {str(e)}")
            
            subreddit_posts = all_posts_from_subreddit[:limit_per_subreddit]
            stats['total_posts'] += len(subreddit_posts)
            
            status_text.text(f"✅ Total {len(subreddit_posts)} posts fetched from r/{subreddit}. Analyzing...")
            
            # Analyze posts
            for post in subreddit_posts:
                try:
                    title = str(post.title).lower()
                    content = str(post.selftext).lower() if hasattr(post, 'selftext') else ""
                    
                    # Store all posts explored
                    post_info = {
                        'title': str(post.title),
                        'url': f"https://reddit.com{post.permalink}",
                        'author': str(post.author) if post.author else "N/A",
                        'subreddit': subreddit,
                        'date': datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'content_preview': str(post.selftext)[:200] if hasattr(post, 'selftext') else "",
                        'matched_keywords': False,
                        'ai_analyzed': False,
                        'is_lead': False
                    }
                    
                    # Keyword filtering
                    matches_keywords = any(keyword.lower() in title or keyword.lower() in content for keyword in keywords)
                    post_info['matched_keywords'] = matches_keywords
                    
                    if matches_keywords:
                        stats['posts_analyzed'] += 1
                        # Prepare analysis prompt
                        analysis_prompt = f"""
ANALYZE THIS POST FOR B2B LEAD QUALIFICATION:

**Post Details:**
Title: {post.title}
Content: {post.selftext[:600] if hasattr(post, 'selftext') else 'No content'}
Subreddit: r/{subreddit}
Author: u/{post.author if post.author else 'N/A'}

**Your Task:**
Determine if this is a potential B2B customer for business intelligence/analytics software.

**Analysis Required:**
1. Is this a BUSINESS need or personal/student project?
2. Are they a decision-maker or influencer in their organization?
3. What specific BI/analytics pain points do they have?
4. Do they mention budget, team size, or enterprise requirements?
5. What red flags exist (if any)?

**Provide Output in This Format:**
Score: [1-10 based on scoring guide]
Business Context: [Is this business or personal? Evidence?]
Decision Authority: [Likely decision-maker? Why/why not?]
Pain Points: [Specific BI/analytics challenges mentioned]
Budget Indicators: [Any mention of budget, paid tools, or willingness to invest?]
Red Flags: [Student/hobby/tutorial-seeking indicators?]
Recommendation: [Should we pursue this lead? Why?]

Be critical and specific. We only want real business opportunities.
"""
                        
                        try:
                            post_info['ai_analyzed'] = True
                            response = lead_tracker_agent.run(analysis_prompt)
                            
                            if response and response.content:
                                response_text = str(response.content)
                                
                                # Extract score
                                score = min_score  # Default
                                if "Score:" in response_text:
                                    try:
                                        score_line = [line for line in response_text.split('\n') if 'Score:' in line][0]
                                        score = int(''.join(filter(str.isdigit, score_line)))
                                    except:
                                        pass
                                
                                post_info['ai_score'] = score
                                post_info['ai_analysis'] = response_text[:400]
                                
                                # Only add if score meets threshold
                                if score >= min_score:
                                    post_info['is_lead'] = True
                                    stats['leads_found'] += 1
                                    lead_data = {
                                        "username": str(post.author) if post.author else "N/A",
                                        "post_title": str(post.title),
                                        "post_url": f"https://reddit.com{post.permalink}",
                                        "post_content": str(post.selftext)[:300] if hasattr(post, 'selftext') else "",
                                        "subreddit": subreddit,
                                        "relevance_score": score,
                                        "identified_needs": [kw for kw in keywords if kw.lower() in title or kw.lower() in content],
                                        "post_date": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                                        "ai_analysis": response_text[:400]
                                    }
                                    all_leads.append(lead_data)
                        except Exception as agent_error:
                            # Fallback to keyword-based lead
                            lead_data = {
                                "username": str(post.author) if post.author else "N/A",
                                "post_title": str(post.title),
                                "post_url": f"https://reddit.com{post.permalink}",
                                "post_content": str(post.selftext)[:300] if hasattr(post, 'selftext') else "",
                                "subreddit": subreddit,
                                "relevance_score": min_score,
                                "identified_needs": [kw for kw in keywords if kw.lower() in title or kw.lower() in content],
                                "post_date": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                                "ai_analysis": "Keyword match - agent analysis unavailable"
                            }
                            all_leads.append(lead_data)
                    
                    # Add post to explored list
                    all_posts_explored.append(post_info)
                
                except Exception as post_error:
                    continue
            
        except Exception as subreddit_error:
            st.warning(f"Error processing r/{subreddit}: {str(subreddit_error)}")
        
        # Update progress
        progress_bar.progress((idx + 1) / total_subreddits)
    
    status_text.text(f"✅ Search completed! Found {stats['leads_found']} leads from {stats['total_posts']} posts.")
    progress_bar.progress(1.0)
    
    return all_leads, all_posts_explored, stats


# Main UI
st.markdown('<div class="main-header">🔍 Reddit Lead Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Find potential customers for your data analytics software on Reddit</div>', unsafe_allow_html=True)

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("🔑 API Credentials", expanded=True):
        openai_api_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key for AI analysis")
        
        st.markdown("**Reddit API Credentials**")
        reddit_client_id = st.text_input("Reddit Client ID", help="From Reddit app preferences")
        reddit_client_secret = st.text_input("Reddit Client Secret", type="password", help="From Reddit app preferences")
        reddit_username = st.text_input("Reddit Username")
        reddit_password = st.text_input("Reddit Password", type="password")
    
    with st.expander("🎯 Search Parameters", expanded=True):
        st.markdown("**Subreddits to Search**")
        default_subreddits = ["dataanalysis", "datascience", "businessintelligence", "analytics", "visualization", "PowerBI", "tableau", "excel"]
        subreddits_input = st.text_area(
            "Enter subreddits (one per line)",
            value="\n".join(default_subreddits),
            height=150
        )
        subreddits = [s.strip() for s in subreddits_input.split("\n") if s.strip()]
        
        st.markdown("**Keywords**")
        default_keywords = [
            "business intelligence",
            "BI dashboard",
            "company needs analytics",
            "business reporting",
            "enterprise analytics",
            "KPI tracking",
            "business metrics",
            "data visualization for business",
            "management reporting",
            "business analysis tool",
            "corporate dashboard",
            "executive reporting"
        ]
        keywords_input = st.text_area(
            "Enter keywords (one per line)",
            value="\n".join(default_keywords),
            height=120
        )
        keywords = [k.strip() for k in keywords_input.split("\n") if k.strip()]
        
        posts_per_subreddit = st.slider("Posts per subreddit", 10, 3000, 50, 10)
        
        if posts_per_subreddit > 1000:
            st.info(f"💡 Fetching {posts_per_subreddit} posts will use multiple strategies (new + top + hot) to bypass Reddit's 1000-post limit per query.")
        min_relevance_score = st.slider("Minimum relevance score", 1, 10, 7, 1)
    
    st.markdown("---")
    search_button = st.button("🚀 Start Search", use_container_width=True)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Leads", "🔍 All Posts", "💾 Export"])

with tab1:
    if not st.session_state.search_completed:
        st.info("👈 Configure your settings in the sidebar and click 'Start Search' to begin!")
        
        # Show configuration preview
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Subreddits", len(subreddits))
            st.metric("Posts per subreddit", posts_per_subreddit)
        with col2:
            st.metric("Keywords", len(keywords))
            st.metric("Min. score", min_relevance_score)
    else:
        # Show results summary
        st.success(f"✅ Found {len(st.session_state.leads)} potential leads!")
        
        if len(st.session_state.leads) > 0:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Leads", len(st.session_state.leads))
            with col2:
                avg_score = sum(l['relevance_score'] for l in st.session_state.leads) / len(st.session_state.leads)
                st.metric("Avg. Score", f"{avg_score:.1f}/10")
            with col3:
                high_quality = len([l for l in st.session_state.leads if l['relevance_score'] >= 8])
                st.metric("High Quality (8+)", high_quality)
            with col4:
                unique_subreddits = len(set(l['subreddit'] for l in st.session_state.leads))
                st.metric("Subreddits", unique_subreddits)
            
            # Charts
            st.subheader("📈 Lead Distribution")
            
            col1, col2 = st.columns(2)
            with col1:
                # By subreddit
                subreddit_counts = {}
                for lead in st.session_state.leads:
                    subreddit_counts[lead['subreddit']] = subreddit_counts.get(lead['subreddit'], 0) + 1
                st.bar_chart(subreddit_counts)
                st.caption("Leads by Subreddit")
            
            with col2:
                # By score
                score_counts = {}
                for lead in st.session_state.leads:
                    score_counts[lead['relevance_score']] = score_counts.get(lead['relevance_score'], 0) + 1
                st.bar_chart(score_counts)
                st.caption("Leads by Relevance Score")

with tab2:
    if len(st.session_state.leads) > 0:
        st.subheader(f"📋 Found {len(st.session_state.leads)} Leads")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_subreddit = st.selectbox(
                "Filter by subreddit",
                ["All"] + list(set(l['subreddit'] for l in st.session_state.leads))
            )
        with col2:
            filter_score = st.selectbox("Filter by score", ["All", "8-10", "7-9", "6-8"])
        with col3:
            sort_by = st.selectbox("Sort by", ["Relevance Score", "Date", "Subreddit"])
        
        # Apply filters
        filtered_leads = st.session_state.leads.copy()
        
        if filter_subreddit != "All":
            filtered_leads = [l for l in filtered_leads if l['subreddit'] == filter_subreddit]
        
        if filter_score == "8-10":
            filtered_leads = [l for l in filtered_leads if l['relevance_score'] >= 8]
        elif filter_score == "7-9":
            filtered_leads = [l for l in filtered_leads if 7 <= l['relevance_score'] <= 9]
        elif filter_score == "6-8":
            filtered_leads = [l for l in filtered_leads if 6 <= l['relevance_score'] <= 8]
        
        # Sort
        if sort_by == "Relevance Score":
            filtered_leads = sorted(filtered_leads, key=lambda x: x['relevance_score'], reverse=True)
        elif sort_by == "Date":
            filtered_leads = sorted(filtered_leads, key=lambda x: x['post_date'], reverse=True)
        elif sort_by == "Subreddit":
            filtered_leads = sorted(filtered_leads, key=lambda x: x['subreddit'])
        
        # Display leads
        for idx, lead in enumerate(filtered_leads, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### {idx}. {lead['post_title']}")
                    st.markdown(f"**👤 User:** u/{lead['username']} | **📍 Subreddit:** r/{lead['subreddit']} | **📅 Date:** {lead['post_date']}")
                    
                    if lead['post_content']:
                        with st.expander("📄 Post Content"):
                            st.write(lead['post_content'])
                    
                    if lead.get('ai_analysis'):
                        with st.expander("🤖 AI Analysis"):
                            st.write(lead['ai_analysis'])
                    
                    st.markdown(f"**🎯 Identified Needs:** {', '.join(lead['identified_needs'])}")
                    st.markdown(f"[🔗 View on Reddit]({lead['post_url']})")
                
                with col2:
                    score_color = "🟢" if lead['relevance_score'] >= 8 else "🟡" if lead['relevance_score'] >= 6 else "🔴"
                    st.markdown(f"### {score_color} {lead['relevance_score']}/10")
                    st.caption("Relevance Score")
                
                st.markdown("---")
    else:
        st.info("No leads found yet. Run a search to find potential customers!")

with tab3:
    if len(st.session_state.all_posts) > 0:
        st.subheader(f"🔍 All Explored Posts ({len(st.session_state.all_posts)} total)")
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Posts", st.session_state.search_stats['total_posts'])
        with col2:
            keyword_matches = len([p for p in st.session_state.all_posts if p['matched_keywords']])
            st.metric("Keyword Matches", keyword_matches)
        with col3:
            ai_analyzed = len([p for p in st.session_state.all_posts if p['ai_analyzed']])
            st.metric("AI Analyzed", ai_analyzed)
        with col4:
            leads_count = len([p for p in st.session_state.all_posts if p['is_lead']])
            st.metric("Became Leads", leads_count)
        
        st.markdown("---")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_subreddit_all = st.selectbox(
                "Filter by subreddit",
                ["All"] + list(set(p['subreddit'] for p in st.session_state.all_posts)),
                key="all_posts_subreddit"
            )
        with col2:
            filter_status = st.selectbox(
                "Filter by status",
                ["All", "Keyword Match", "AI Analyzed", "Leads Only", "Non-Leads"]
            )
        with col3:
            sort_by_all = st.selectbox("Sort by", ["Date", "Score", "Comments"], key="all_posts_sort")
        
        # Apply filters
        filtered_all_posts = st.session_state.all_posts.copy()
        
        if filter_subreddit_all != "All":
            filtered_all_posts = [p for p in filtered_all_posts if p['subreddit'] == filter_subreddit_all]
        
        if filter_status == "Keyword Match":
            filtered_all_posts = [p for p in filtered_all_posts if p['matched_keywords']]
        elif filter_status == "AI Analyzed":
            filtered_all_posts = [p for p in filtered_all_posts if p['ai_analyzed']]
        elif filter_status == "Leads Only":
            filtered_all_posts = [p for p in filtered_all_posts if p['is_lead']]
        elif filter_status == "Non-Leads":
            filtered_all_posts = [p for p in filtered_all_posts if not p['is_lead']]
        
        # Sort
        if sort_by_all == "Date":
            filtered_all_posts = sorted(filtered_all_posts, key=lambda x: x['date'], reverse=True)
        elif sort_by_all == "Score":
            filtered_all_posts = sorted(filtered_all_posts, key=lambda x: x['score'], reverse=True)
        elif sort_by_all == "Comments":
            filtered_all_posts = sorted(filtered_all_posts, key=lambda x: x['num_comments'], reverse=True)
        
        st.info(f"Showing {len(filtered_all_posts)} posts")
        
        # Display posts in a table format
        for idx, post in enumerate(filtered_all_posts, 1):
            with st.expander(f"{idx}. {post['title'][:80]}... {'🎯' if post['is_lead'] else '✅' if post['matched_keywords'] else ''}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**👤 Author:** u/{post['author']}")
                    st.markdown(f"**📍 Subreddit:** r/{post['subreddit']}")
                    st.markdown(f"**📅 Date:** {post['date']}")
                    
                    if post['content_preview']:
                        st.markdown(f"**📄 Content Preview:**")
                        st.text(post['content_preview'])
                    
                    st.markdown(f"[🔗 View on Reddit]({post['url']})")
                    
                    if post.get('ai_analysis'):
                        st.markdown("**🤖 AI Analysis:**")
                        st.info(post['ai_analysis'])
                
                with col2:
                    st.metric("Reddit Score", post['score'])
                    st.metric("Comments", post['num_comments'])
                    
                    if post.get('ai_score'):
                        st.metric("AI Score", f"{post['ai_score']}/10")
                    
                    status_badges = []
                    if post['matched_keywords']:
                        status_badges.append("🔑 Keywords")
                    if post['ai_analyzed']:
                        status_badges.append("🤖 AI Analyzed")
                    if post['is_lead']:
                        status_badges.append("🎯 Lead")
                    
                    if status_badges:
                        st.markdown("**Status:**")
                        for badge in status_badges:
                            st.markdown(badge)
    else:
        st.info("No posts explored yet. Run a search to see all posts!")

with tab4:
    if len(st.session_state.leads) > 0:
        st.subheader("💾 Export Leads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as JSON
            json_data = json.dumps(st.session_state.leads, indent=4)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"reddit_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export as CSV
            df = pd.DataFrame(st.session_state.leads)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"reddit_leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        st.subheader("📋 Preview")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data to export. Run a search first!")

# Handle search button click
if search_button:
    # Validate inputs
    if not all([openai_api_key, reddit_client_id, reddit_client_secret, reddit_username, reddit_password]):
        st.error("❌ Please fill in all API credentials!")
    elif len(subreddits) == 0:
        st.error("❌ Please enter at least one subreddit!")
    elif len(keywords) == 0:
        st.error("❌ Please enter at least one keyword!")
    else:
        with st.spinner("🔍 Searching Reddit for leads..."):
            try:
                leads, all_posts, stats = track_leads_function(
                    reddit_client_id=reddit_client_id,
                    reddit_client_secret=reddit_client_secret,
                    reddit_username=reddit_username,
                    reddit_password=reddit_password,
                    openai_api_key=openai_api_key,
                    subreddits=subreddits,
                    keywords=keywords,
                    limit_per_subreddit=posts_per_subreddit,
                    min_score=min_relevance_score
                )
                
                st.session_state.leads = leads
                st.session_state.all_posts = all_posts
                st.session_state.search_stats = stats
                st.session_state.search_completed = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Agno Framework | <a href='https://docs.agno.com' target='_blank'>Documentation</a></p>
</div>
""", unsafe_allow_html=True)

