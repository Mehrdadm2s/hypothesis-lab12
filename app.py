import os
import json
import gradio as gr
from huggingface_hub import InferenceClient


SYSTEM_PROMPT = """
تو موتور تحلیل «Personal Hypothesis Lab» هستی.

وظیفه تو این نیست که به کاربر بگویی ایده‌اش درست است.
وظیفه تو تبدیل یک فکر خام به یک فرضیه قابل بررسی است.

قوانین:

1. فکر خام کاربر را از تفسیر خودت جدا کن.
2. ادعای اصلی را استخراج کن.
3. مشخص کن آیا ادعا مشاهده، سؤال، فرضیه، پیش‌بینی یا نظریه است.
4. بهترین استدلال موافق را بنویس.
5. بهترین استدلال مخالف را بنویس.
6. پیش‌بینی قابل آزمون تولید کن.
7. مشخص کن چه مشاهده‌ای می‌تواند فرضیه را تضعیف یا رد کند.
8. اگر ایده فاقد آزمون تجربی مستقیم است، صادقانه بگو.
9. میزان اطمینان را بین 0 تا 100 درصد تخمین بزن.
10. «جالب بودن» را با «درست بودن» اشتباه نگیر.
11. اگر رابطه‌ای صرفاً تداعی یا استعاره است، آن را رابطه علمی معرفی نکن.
12. اگر اطلاعات کافی نیست، بگو «اطلاعات کافی نداریم».
13. هرگز برای خوشحال کردن کاربر ادعای او را تأیید نکن.

پاسخ را با ساختار زیر بده:

نوع ایده:
ادعای اصلی:
تفسیر محتمل:
شواهد موافق:
شواهد مخالف:
پیش‌بینی:
چه چیزی می‌تواند آن را رد کند:
فرضیه‌های جایگزین:
اطمینان اولیه:
اهمیت احتمالی:
آزمایش پیشنهادی:
"""


def analyze_idea(idea: str) -> str:
    """
    Analyze a user's raw idea and transform it into a structured,
    testable hypothesis.
    """
    if not idea or not idea.strip():
        return "لطفاً ابتدا یک ایده یا مشاهده وارد کن."

    token = os.environ.get("HF_TOKEN")

    if not token:
        return (
            "برنامه ساخته شده، اما هنوز HF_TOKEN در تنظیمات Space قرار نگرفته است.\n\n"
            "مرحله بعدی این است که توکن Hugging Face را به‌عنوان Secret اضافه کنیم."
        )

    try:
        client = InferenceClient(
            token=token,
            provider="auto"
        )

        prompt = f"""
{SYSTEM_PROMPT}

فکر خام کاربر:

{idea}

اکنون آن را تحلیل کن.
"""

        response = client.chat_completion(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1800,
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"خطا در اتصال به مدل:\n\n{str(e)}"


with gr.Blocks(title="Personal Hypothesis Lab") as demo:

    gr.Markdown(
        """
        # 🧠 Personal Hypothesis Lab

        ### آزمایشگاه شخصی ایده، فرضیه و پیش‌بینی

        یک فکر خام، مشاهده یا سؤال را وارد کن.
        سیستم تلاش می‌کند آن را به یک فرضیه قابل بررسی تبدیل کند.
        """
    )

    idea_input = gr.Textbox(
        label="فکر خام / مشاهده / سؤال",
        placeholder=(
            "مثلاً: شاید زبان شکل خاصی از تعامل یک سیستم دارای حافظه با محیط باشد..."
        ),
        lines=8,
    )

    analyze_button = gr.Button(
        "🔬 تبدیل به فرضیه و تحلیل"
    )

    result_output = gr.Markdown(
        label="تحلیل"
    )

    analyze_button.click(
        fn=analyze_idea,
        inputs=idea_input,
        outputs=result_output,
    )


if __name__ == "__main__":
    demo.launch()