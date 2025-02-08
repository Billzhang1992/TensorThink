import datetime
from dataclasses import dataclass
from typing import Any, List, Dict, Union, Optional
import logfire
from pydantic import BaseModel, Field
from rich.prompt import Prompt
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits
from httpx import AsyncClient
from dotenv import load_dotenv
import os

logfire.configure(send_to_logfire='if-token-present')
load_dotenv()


class CompanyDetails(BaseModel):
    """Details of the target company."""
    company_name: str
    establishment_date: datetime.date = Field(description='成立日期')
    registered_address: str = Field(description='注册地址')
    business_scope: str = Field(description='经营范围')
    industry: str = Field(description='所属行业')

class NoCompanyFound(BaseModel):
    """When no valid company is found."""


# ---------------------------
# 共享上下文数据定义
# ---------------------------
@dataclass
class SearchCompanyDeps:
    login_user_id: str       # 当前登录用户 id
    req_query: str           # 用户初始 query
    client: AsyncClient      # 用于 HTTP 请求的 client
    api_key: Optional[str]   # 请求 API 所需的 key


@dataclass
class SearchCompanyEmployeeDeps:
    login_user_id: str       # 当前登录用户 id
    req_company_id: str           # 用户初始 query
    client: AsyncClient      # 用于 HTTP 请求的 client
    api_key: Optional[str]   # 请求 API 所需的 key


# 最终的结果通知agent的上下文数据结构
@dataclass
class ResultNoticeDeps:
    login_user_id: str            # 目标通知用户 id
    result_data: Dict[str, Any]  # 结果数据列表


# Dummy 数据：潜在目标公司信息（模拟 API 响应）
alibaba_xiaoman_data = {
    "companies": [
        {
            "company_id": "TV001",
            "company_name": "TechVision Solutions Inc.",
            "registered_capital": "$50 million",
            "operating_status": "Active",
            "establishment_date": "2015-03-18",
            "registered_address": "123 Innovation Drive, Silicon Valley, CA 94025",
            "unified_social_credit_code": "US440300MA5F7XY2K8",
            "company_type": "Corporation",
            "industry": "Electronics Distribution",
            "business_scope": "Distribution and retail of Apple brand electronic products and accessories; electronic product technology development and sales; domestic trade; import and export business."
        },
        {
            "company_id": "SL002",
            "company_name": "SmartLink Digital Technologies Inc.",
            "registered_capital": "$30 million",
            "operating_status": "Active",
            "establishment_date": "2016-06-25",
            "registered_address": "456 Tech Park Avenue, Boston, MA 02110",
            "unified_social_credit_code": "US310115MA1K3B7X9N",
            "company_type": "Corporation",
            "industry": "Electronics Distribution",
            "business_scope": "Sales of Apple brand accessories, smart devices and peripheral products; electronic product technical consulting; business information consulting; e-commerce."
        },
        {
            "company_id": "GT003",
            "company_name": "GreenTech Environmental Solutions Inc.",
            "registered_capital": "$80 million",
            "operating_status": "Active",
            "establishment_date": "2018-09-12",
            "registered_address": "789 Eco Boulevard, Seattle, WA 98104",
            "unified_social_credit_code": "US110108MA1HK9P4X2",
            "company_type": "Corporation",
            "industry": "Environmental Technology",
            "business_scope": "Environmental equipment R&D, production and sales; environmental management technology development; environmental engineering design and construction; environmental consulting services."
        }
    ]
}
# Dummy 数据：公司员工信息（模拟 API 响应）
tech_vision_data = {
    "company_id": "TV001",
    "company_name": "TechVision Solutions Inc.",
    "legal_representative": {
        "name": "Michael Anderson",
        "position": "Chairman and CEO",
        "phone": "+1-650-888-8888",
        "email": "michael.anderson@techvision.com",
        "linkedin": "linkedin.com/in/michael-anderson"
    },
    "procurement_managers": [
        {
            "name": "Robert Strong",
            "position": "Procurement Director",
            "phone": "+1-650-123-4567",
            "email": "robert.strong@techvision.com",
            "linkedin": "linkedin.com/in/robert-strong"
        },
        {
            "name": "Sarah Williams",
            "position": "Senior Procurement Manager",
            "phone": "+1-650-234-5678",
            "email": "sarah.williams@techvision.com",
            "linkedin": "linkedin.com/in/sarah-williams"
        }
    ]
}

smart_link_data = {
    "company_id": "SL002",
    "company_name": "SmartLink Digital Technologies Inc.",
    "legal_representative": {
        "name": "David Thompson",
        "position": "Chairman",
        "phone": "+1-617-999-9999",
        "email": "david.thompson@smartlink.com",
        "linkedin": "linkedin.com/in/david-thompson"
    },
    "procurement_managers": [
        {
            "name": "James Wilson",
            "position": "Procurement Department Manager",
            "phone": "+1-617-456-7890",
            "email": "james.wilson@smartlink.com",
            "linkedin": "linkedin.com/in/james-wilson"
        },
        {
            "name": "Emily Davis",
            "position": "Procurement Supervisor",
            "phone": "+1-617-567-8901",
            "email": "emily.davis@smartlink.com",
            "linkedin": "linkedin.com/in/emily-davis"
        },
        {
            "name": "Christopher Brown",
            "position": "Procurement Specialist",
            "phone": "+1-617-678-9012",
            "email": "christopher.brown@smartlink.com",
            "linkedin": "linkedin.com/in/christopher-brown"
        }
    ]
}

green_tech_data = {
    "company_id": "GT003",
    "company_name": "GreenTech Environmental Solutions Inc.",
    "legal_representative": {
        "name": "Jennifer Parker",
        "position": "Chairman and CEO",
        "phone": "+1-206-777-7777",
        "email": "jennifer.parker@greentech.com",
        "linkedin": "linkedin.com/in/jennifer-parker"
    },
    "procurement_managers": [
        {
            "name": "Thomas Miller",
            "position": "Procurement Director",
            "phone": "+1-206-789-0123",
            "email": "thomas.miller@greentech.com",
            "linkedin": "linkedin.com/in/thomas-miller"
        },
        {
            "name": "Rachel Johnson",
            "position": "Procurement Manager",
            "phone": "+1-206-890-1234",
            "email": "rachel.johnson@greentech.com",
            "linkedin": "linkedin.com/in/rachel-johnson"
        }
    ]
}

# 将所有公司数据合并到一个列表中
linkedin_data = {
    "companies": [tech_vision_data, smart_link_data, green_tech_data]
}

# Dummy 推送通知返回值
success_result_notice = {
    "status_code": 200,
    "message": "success"
}
failed_result_notice = {
    "status_code": 500,
    "message": "failed"
}


# ---------------------------
# Model & Agent 初始化
# ---------------------------
model = OpenAIModel(
    model_name='gpt-4o',
    api_key=os.getenv('OPENAI_API_KEY')
)

# ---------------------------
# 搜索公司代理
# ---------------------------
search_company_agent = Agent[SearchCompanyDeps, list[CompanyDetails]|NoCompanyFound](
    model,
    result_type=list[CompanyDetails]|NoCompanyFound,
    retries=4,
    system_prompt=(
        'Your job is to search and analyze company information based on user queries. Follow these three steps:\n\n'
        '1. Use the `search_company` tool to find and gather basic information about relevant companies\n'
        '2. Use the `filter_company_data` tool to further filter and analyze the results based on relevance.\n'
        '3. Use the `extract_company_data` tool to extract the filtered results\n'
    ),
)

# 筛选企业信息的Agent（如果需要对文本进行进一步处理，可在此实现）
filter_company_info_agent = Agent(
    model,
    system_prompt='Your job is to further filter and analyze relevant information from the data based on the user query.'
)

# 辅助的抽取Agent（如果需要对文本进行进一步处理，可在此实现）
extract_company_info_agent = Agent(
    model,
    result_type=list[CompanyDetails]|NoCompanyFound,
    system_prompt='Extract all the company details from the given text'
)


@search_company_agent.tool
async def search_company(ctx: RunContext[SearchCompanyDeps], user_query: str) -> str:
    """
    根据用户 query 搜索公司信息。
    
    Returns:
        - 成功: {"status": "success", "data": [...公司列表...]}
        - 失败: {"status": "error", "message": "错误信息"}
    """
    if ctx.deps.api_key is None:
        return {"status": "error", "message": "API key 不存在"}

    params = {
        'query': ctx.deps.req_query,
        'api_key': ctx.deps.api_key
    }
    with logfire.span('calling search_company API', params=params) as span:
        # 1) 若为真实场景，请使用:
        #    r = await ctx.deps.client.get('https://zft.com/example/search_company', params=params)
        #    r.raise_for_status()
        #    data = r.json()
        #
        # 2) 这里使用测试假数据:
        data = alibaba_xiaoman_data
        span.set_attribute('mock_response', data)

    if data and "companies" in data:
        companies = data.get("companies", [])
        if len(companies) > 0:
            return str(companies)
        return ""
    else:
        # 如果没找到数据，可选择抛出异常让Agent重试，或者直接返回错误信息
        return ""


@search_company_agent.tool
async def filter_company_data(ctx: RunContext[SearchCompanyDeps], all_related_company_data: str) -> str:
    """
    对搜索公司所返回的数据进行进一步的筛选。
    """
    use_query = ctx.deps.req_query

    result = await filter_company_info_agent.run(
        f'Based on the query "{use_query}" and the following company data:\n{all_related_company_data}\n\nAnalyze and filter the companies. Focus on their business scope, industry, and relevance to the query. Return only the most relevant companies that match the search criteria.',
        usage=ctx.usage
    )
    return result.data

@search_company_agent.tool
async def extract_company_data(ctx: RunContext[SearchCompanyDeps], filtered_company_data: str) -> list[CompanyDetails]|NoCompanyFound:
    """
    对搜索公司所返回的数据进行进一步解析或清洗。
    如无需要，可不做复杂处理，或直接返回。
    """
    # 如果需要使用Agent对文本做抽取，可以在这里执行:
    # SearchCompanyDeps

    use_query = ctx.deps.req_query
    result = await extract_company_info_agent.run(
        f'{filtered_company_data}',
        usage=ctx.usage
    )
    return result.data

    # # 当前示例只是直接返回原数据:
    # logfire.info("No additional extraction logic - returning company data directly.")
    # return {"extracted_companies": all_related_company_data}


# ---------------------------
# 主函数
# ---------------------------

usage_limits = UsageLimits(request_limit=15)

async def main():
    async with AsyncClient() as client:
        # 此处 api_key 为模拟值，实际使用时请替换为有效的 API key
        api_key = "sk_123"
        req_query = 'Apple accessories distributor'
        login_user_id = "1000001"

        deps = SearchCompanyDeps(
            login_user_id=login_user_id, 
            client=client,
            req_query=req_query, 
            api_key=api_key
        )

        message_history: Optional[List[ModelMessage]] = None
        usage: Usage = Usage()
        # 运行搜索 agent，直到找到匹配公司或确定无结果

        result = await search_company_agent.run(
            f'Find me information about {deps.req_query}',
            deps=deps,
            usage=usage,
            message_history=message_history,
            usage_limits=usage_limits,
        )
        print(result.data)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

