from pydantic import BaseModel


class Example(BaseModel):
    """
    A class to represent a few-shot example.
    """
    query: str
    passage: str
    label: str
    explanation: str
    factuality: str
    information_density: str
    commonsense: str
    textual_description: str


# a list of few-shot examples
list_examples = [
    Example(
        query="what is a bak file",
        passage="The easiest way to open a BAK file is to double-click on it and let your PC decide which default application should open the file. "
                "If no program opens the BAK file then you probably don't have an application installed that can view and/or edit BAK files."
                "1 Open the BAK file in its default program and choose to save the open file as another file format. 2 Use a File Conversion Online Service or Software Program to convert the BAK file to",
        label="not relevant",
        explanation="Only explains how to open a bak file.",
        factuality="Text is factual, but does not contain definitions.",
        information_density="Moderate",
        commonsense="Since it is called BAK file, it has to be a document on the PC, which can be opened with a mouse click and can be edited.",
        textual_description="tutorial, short sentences"
    ),
    Example(
        query="what is kuchen",
        passage="Grandmother's Kuchen. Giora Shimoni. Kuchen means cake in German, and refers to a variety of cakes."
                " This kuchen, which was my grandmother's recipe, is a coffee cake with veins and pockets of baked-in cinnamon and sugar.",
        label="relevant",
        explanation="Contains the direct translation of Kuchen.",
        factuality="Text is factual, but also contains personal information.",
        information_density="High",
        commonsense="Kuchen is a german word. Its translation is cake. Since it is food a recipe can exist.",
        textual_description="story, personal information"
    ),
    Example(
        query="does human hair stop squirrels",
        passage="We have been feeding our back yard squirrels for the fall and winter and we noticed that a few of them have missing fur."
                "One has a patch missing down his back and under both arms. Also another has some missing on his whole chest."
                "They are all eating and seem to have a good appetite.",
        label="not relevant",
        explanation="Text is only about squirrel fur.",
        factuality="Text is not very factual, contains more personal observation.",
        information_density="Low",
        commonsense="Neither human hair nor any synonym are mentioned, therefore query can not be answered.",
        textual_description="story, personal observation, question-like"
    ),
    Example(
        query="what is a cost variance",
        passage="Cost variance is, in short, the difference between the budgeted cost and the actual cost on your project at any one point."
                " I want to know what the formula to calculate the cost variance is. It'll be great if someone can give me the general"
                "formula as well as an example applying the formula for calculating the cost variance of a project.",
        label="not relevant",
        explanation="Text provides a basic definition but shifts focus to a query about calculation.",
        factuality="Factually is correct but incomplete.",
        information_density="Low",
        commonsense="The passage's question format suggests a lack of authoritative information.",
        textual_description="brief, inquiry-focused."
    ),
    Example(
        query="what is a cost variance",
        passage="A cost variance is the difference between the cost actually incurred and the budgeted or planned amount"
                " of cost that should have been incurred. Cost variances are most commonly tracked for expense line items,"
                " but can also be tracked at the job or project level, as long as there is a budget or standard against"
                " which it can be calculated. These variances form a standard part of many management reporting systems.",
        label="relevant",
        explanation="Offers a detailed explanation of cost variance and its practical applications",
        factuality="Factual and informative; provides a clear and accurate description of cost variance.",
        information_density="High",
        commonsense="Cost variance is mentioned, variance generally stands for difference, which is mentioned and also its use case for project and reporting.",
        textual_description="short sentences, descriptive, concise, clear"
    ),
]
list_examples_dict = [example.dict() for example in list_examples]
