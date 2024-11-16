from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import openai

class MallCustomerRBA:
    def __init__(self):


        # 设置环境变量
        os.environ["http_proxy"] = "http://127.0.0.1:7890"
        os.environ["https_proxy"] = "http://127.0.0.1:7890"
        os.environ[
            "OPENAI_API_KEY"] = "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"

        # 设置API密钥
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        """初始化RBA系统"""
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.3
        )

        # 1. 数据分析师
        self.analyst = Agent(
            role='Data Analyst',
            goal='Analyze customer data and create visualizations',
            backstory="""You are an expert data analyst specializing in customer segmentation.
            Analyze the preprocessed customer data and provide insights with visualizations.""",
            llm=self.llm,
            verbose=True
        )

        # 2. 营销专家
        self.marketer = Agent(
            role='Marketing Expert',
            goal='Develop targeted marketing strategies with ROI projections',
            backstory="""You are a marketing expert who develops targeted strategies.
            Create specific marketing recommendations with ROI estimates for each segment.""",
            llm=self.llm,
            verbose=True
        )

    def create_visualizations(self, data: pd.DataFrame) -> List[str]:
        """创建可视化"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_files = []

        # 1. 客户分布散点图
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data['Annual Income (k$)'],
                              data['Spending Score (1-100)'],
                              c=data['Cluster'],
                              cmap='viridis')
        plt.title('Customer Segments Distribution')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score')
        plt.colorbar(scatter, label='Cluster')
        filename = f'segment_distribution_{timestamp}.png'
        plt.savefig(filename)
        viz_files.append(filename)
        plt.close()

        # 2. 年龄分布箱型图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y='Age', data=data)
        plt.title('Age Distribution by Segment')
        filename = f'age_distribution_{timestamp}.png'
        plt.savefig(filename)
        viz_files.append(filename)
        plt.close()

        # 3. 收入-支出热力图
        plt.figure(figsize=(10, 6))
        pivot_table = pd.crosstab(
            pd.qcut(data['Annual Income (k$)'], 5),
            pd.qcut(data['Spending Score (1-100)'], 5)
        )
        sns.heatmap(pivot_table, annot=True, cmap='YlOrRd')
        plt.title('Income vs Spending Distribution')
        filename = f'income_spending_heatmap_{timestamp}.png'
        plt.savefig(filename)
        viz_files.append(filename)
        plt.close()

        return viz_files

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        df = data.copy()
        scaler = StandardScaler()
        numeric_cols = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        return df

    def get_cluster_stats(self, data: pd.DataFrame) -> Dict:
        """计算客户群统计信息"""
        cluster_stats = {}
        for cluster in data['Cluster'].unique():
            cluster_data = data[data['Cluster'] == cluster]

            # 计算年消费潜力
            avg_income = cluster_data['Annual Income (k$)'].mean()
            avg_spending_score = cluster_data['Spending Score (1-100)'].mean()
            potential_annual_spending = avg_income * (avg_spending_score / 100) * 1000  # 估算年消费潜力

            cluster_stats[f'Cluster_{cluster}'] = {
                'size': len(cluster_data),
                'age': {
                    'mean': round(float(cluster_data['Age'].mean()), 2),
                    'min': int(cluster_data['Age'].min()),
                    'max': int(cluster_data['Age'].max())
                },
                'income': {
                    'mean': round(float(cluster_data['Annual Income (k$)'].mean()), 2),
                    'min': int(cluster_data['Annual Income (k$)'].min()),
                    'max': int(cluster_data['Annual Income (k$)'].max())
                },
                'spending': {
                    'mean': round(float(cluster_data['Spending Score (1-100)'].mean()), 2),
                    'min': int(cluster_data['Spending Score (1-100)'].min()),
                    'max': int(cluster_data['Spending Score (1-100)'].max())
                },
                'potential_annual_spending': round(potential_annual_spending, 2)
            }
        return cluster_stats

    def analyze_customers(self, data: pd.DataFrame) -> Dict:
        """执行客户分析流程"""
        # 数据预处理
        processed_data = self.preprocess_data(data)

        # 执行聚类
        X = processed_data[['Annual Income (k$)', 'Spending Score (1-100)']]
        kmeans = KMeans(n_clusters=5, random_state=42)
        data['Cluster'] = kmeans.fit_predict(X)

        # 创建可视化
        viz_files = self.create_visualizations(data)

        # 获取统计信息
        cluster_stats = self.get_cluster_stats(data)

        # 1. 分析任务
        analysis_task = Task(
            description=f"""
            Analyze the customer segments with visualizations:
            Cluster Statistics: {json.dumps(cluster_stats, indent=2)}
            Visualization files: {viz_files}

            Provide segmentation insights in JSON format:
            {{
                "segments": {{
                    "segment_1": {{
                        "name": "segment name",
                        "profile": "key characteristics",
                        "potential_value": "annual spending potential",
                        "visualization_insights": "insights from visual analysis"
                    }}
                }},
                "key_insights": ["insight1", "insight2"]
            }}
            """,
            agent=self.analyst
        )

        # 2. 营销任务
        marketing_task = Task(
            description=f"""
            Based on the segments and their potential spending:
            {json.dumps(cluster_stats, indent=2)}

            Provide marketing strategies with ROI estimates in JSON format:
            {{
                "segment_strategies": {{
                    "segment_1": {{
                        "approach": "main strategy",
                        "channels": ["channel1", "channel2"],
                        "tactics": ["tactic1", "tactic2"],
                        "roi_analysis": {{
                            "estimated_cost": "estimated marketing cost",
                            "expected_revenue": "projected revenue",
                            "roi_percentage": "expected ROI percentage",
                            "payback_period": "estimated months to positive ROI"
                        }}
                    }}
                }},
                "priorities": ["priority1", "priority2"],
                "budget_allocation": {{
                    "segment_1": "xx%"
                }}
            }}
            """,
            agent=self.marketer
        )

        # 创建Crew并执行任务
        crew = Crew(
            agents=[self.analyst, self.marketer],
            tasks=[analysis_task, marketing_task],
            process=Process.sequential,
            verbose=True
        )

        results = crew.kickoff()

        return {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'visualizations': viz_files,
            'cluster_statistics': cluster_stats,
            'analysis_results': results
        }


def main():
    """主函数"""
    try:
        # 读取数据
        df = pd.read_csv('Mall_Customers.csv')
        print(f"Loaded data with shape: {df.shape}")

        # 创建RBA系统并运行分析
        rba_system = MallCustomerRBA()
        results = rba_system.analyze_customers(df)

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'mall_analysis_results_{timestamp}.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print("\n=== Analysis Completed Successfully ===")
        print(f"\nResults saved to '{output_file}'")
        print("\nGenerated visualizations:")
        for viz in results['visualizations']:
            print(f"- {viz}")

        return results

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()