from data_crawler import WeiboHotCrawler, WeiboTextCrawler, ZhihuSearchCrawler
from dotenv import load_dotenv
import json
from datetime import datetime
from pathlib import Path

load_dotenv()

# 使用绝对路径，避免工作目录不同导致的问题
OUTPUT_DIR = Path(__file__).parent.parent / "html/data"  # GitHub Pages 会读取这个目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 调试：打印输出路径
print(f"[INFO] 输出目录: {OUTPUT_DIR.resolve()}")


def analyze_sentiment(text):
    """情感分析（简单规则）"""
    positive_words = ["好", "棒", "喜欢", "爱", "赞", "优秀", "精彩", "推荐", "支持", "不错", "可以"]
    negative_words = ["差", "坏", "讨厌", "恨", "烂", "垃圾", "失望", "糟糕", "反对", "不好"]

    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)

    if pos_count > neg_count:
        return "positive", pos_count / (pos_count + neg_count + 1)
    elif neg_count > pos_count:
        return "negative", neg_count / (pos_count + neg_count + 1)
    else:
        return "neutral", 0.5


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("="*50)
    print("NLP 公众意见分析 - 数据采集")
    print("="*50)

    # ===== 1. 抓取微博热搜 =====
    print("\n[1/4] 抓取微博热搜...")
    hot_crawler = WeiboHotCrawler()
    weibo_hot = hot_crawler.crawl()
    print(f"  ✓ 抓取到 {len(weibo_hot)} 条热搜")

    # ===== 2. 抓取微博搜索内容（取前3个热搜）=====
    print("\n[2/4] 抓取微博搜索内容...")
    text_crawler = WeiboTextCrawler()
    weibo_posts = []

    for item in weibo_hot[:3]:
        keyword = item['word']
        print(f"  搜索: {keyword}")
        results = text_crawler.crawl(keyword, max_pages=2)
        weibo_posts.extend(results)
        print(f"    找到 {len(results)} 条")

    print(f"  ✓ 共抓取 {len(weibo_posts)} 条微博")

    # ===== 3. 抓取知乎讨论 =====
    print("\n[3/4] 抓取知乎讨论...")
    zhihu_crawler = ZhihuSearchCrawler()
    zhihu_data = []

    # 用前 5 个热搜在知乎搜索
    for item in weibo_hot[:5]:
        keyword = item['word']
        print(f"  搜索: {keyword}")
        results = zhihu_crawler.crawl(keyword, max_results=3)
        zhihu_data.extend(results)

    print(f"  ✓ 抓取到 {len(zhihu_data)} 条知乎内容")

    # ===== 4. 情感分析 =====
    print("\n[4/4] 情感分析...")
    for item in zhihu_data:
        sentiment, score = analyze_sentiment(item['content'])
        item['sentiment'] = sentiment
        item['sentiment_score'] = score

    # 对微博内容也做情感分析
    for item in weibo_posts:
        sentiment, score = analyze_sentiment(item['text'])
        item['sentiment'] = sentiment
        item['sentiment_score'] = score

    print(f"  ✓ 分析完成")

    # ===== 5. 生成 JSON 数据 =====
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "weibo_hot_count": len(weibo_hot),
            "weibo_posts_count": len(weibo_posts),
            "zhihu_count": len(zhihu_data),
        },
        "weibo_hot": weibo_hot[:10],  # 只存前10条
        "weibo_posts": weibo_posts[:20],  # 只存前20条
        "zhihu": zhihu_data,
        "sentiment_summary": {
            "positive": sum(1 for x in zhihu_data if x.get('sentiment') == 'positive') + sum(1 for x in weibo_posts if x.get('sentiment') == 'positive'),
            "negative": sum(1 for x in zhihu_data if x.get('sentiment') == 'negative') + sum(1 for x in weibo_posts if x.get('sentiment') == 'negative'),
            "neutral": sum(1 for x in zhihu_data if x.get('sentiment') == 'neutral') + sum(1 for x in weibo_posts if x.get('sentiment') == 'neutral'),
        }
    }

    # 保存到 docs/data/ 目录
    output_file = OUTPUT_DIR / "analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 数据已保存到 {output_file}")

    # 打印摘要
    print("\n" + "="*50)
    print("数据摘要:")
    print("="*50)
    print(f"微博热搜: {len(weibo_hot)} 条")
    print(f"微博内容: {len(weibo_posts)} 条")
    print(f"知乎内容: {len(zhihu_data)} 条")
    print(f"情感分布: 正面 {output_data['sentiment_summary']['positive']} | "
          f"负面 {output_data['sentiment_summary']['negative']} | "
          f"中性 {output_data['sentiment_summary']['neutral']}")
    print("="*50)

    # 打印热搜列表
    print("\n热搜 Top 5:")
    for item in weibo_hot[:5]:
        label = item.get('label_name', '')
        print(f"  {item['rank']}. {item['word']} {f'[{label}]' if label else ''}")


if __name__ == '__main__':
    main()
