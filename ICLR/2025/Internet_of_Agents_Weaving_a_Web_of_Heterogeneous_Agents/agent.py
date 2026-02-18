"""
Internet of Agents: 异构智能体协作框架
"""

import asyncio

class IoAAgent:
    def __init__(self, agent_id, capabilities):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.inbox = asyncio.Queue()
        self.team = []
        
    async def send_message(self, recipient, message):
        """发送消息给其他智能体"""
        await recipient.inbox.put({
            'sender': self.agent_id,
            'content': message
        })
        
    async def receive_messages(self):
        """接收消息"""
        messages = []
        while not self.inbox.empty():
            msg = await self.inbox.get()
            messages.append(msg)
        return messages
    
    async def form_team(self, task_requirements):
        """形成团队"""
        available = [a for a in self.team if self._can_help(a, task_requirements)]
        return available[:3]  # 最多3个队友
        
    def _can_help(self, agent, requirements):
        return any(cap in agent.capabilities for cap in requirements)

class IoASystem:
    def __init__(self):
        self.agents = {}
        
    def register_agent(self, agent):
        self.agents[agent.agent_id] = agent
        
    async def distribute_task(self, task):
        """分发任务"""
        teams = {}
        for agent in self.agents.values():
            team = await agent.form_team(task['requirements'])
            teams[agent.agent_id] = team
        return teams
