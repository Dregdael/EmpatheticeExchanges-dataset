# EmpatheticExchanges

This is a repository for the article "EmpatheticExchanges: Towards Understanding the Cues for Empathy in Dyadic Conversations." in IEEE Access.

The main goal for this repository is to provide access to our novel dataset which presents a series of empathetic conversations separated into exchanges. These exchanges present a score for behavioral empathy using our novel *3ES* and *5ES* metrics. This can be used for classification of empathy in textual exchanges using various classifiers. Our work is based on EmpatheticDialogues by Rashkin et al. [1]

In addition to the empathy metrics, each dyadic exchange presents the following features:

    • *Conv_id*: An unique identifier for the conversation in the database. This also corresponds to the identifier in EmpatheticDialogues.
    • *Context*: A description of an emotional label given to the conversation.
    • *Prompt*: The description of the emotional situation the defines the conversation.
    • *speaker_utterance*: The utterance by the first person involved in the exchange.
    • *listener_utterance*: The utterance by the second person involved in the exchange. We measure the displayed behavioral empathy of this utterance based on the utterance by the speaker.
    • *exchange_number*: The position of the exchange in the conversation.
    • *s_sentiment*: Values for negative, neutral, and positive sentiment for the speaker utterance. Given in range [0,1].
    • *l_sentiment*: Values for negative, neutral, and positive sentiment for the listener utterance. Given in range [0,1].
    • *EPITOME values*: Values for Emotional Reaction, Interpretation, and Exploration communication mechanisms [2] of the exchange. Given in range [0,2].
    • *VAD vectors_speaker*: Values for Valence, Arousal, and Dominance for speaker utterance. Given in range [-1,1].
    • *VAD vectors_listener*: Values for Valence, Arousal, and Dominance for listener utterance. Given in range [-1,1].
    • *s_word_len*: Length of speaker utterance.
    • *l_word_len*: Length of listener utterance.
    • *Empathetic intent scores*: Values for possible empathetic intents of the listener response. Given in range [0,1].
    • *Empathetic intent scores*: Values for possible empathetic intents of the listener response [3]. Given in range [0,1].
    • *Emotions*: Emotion labels for both utterances.
    • *Mimicry*: Binary label defining whether emotional mimicry is present in the exchange.


We provide the original dataset and the code to replicate the work found on the paper.

In addition, we provide access to an enhanced dataset that contains more conversations labeled with our empathy metric, this one is called EmpatheticExchanges_extended.

[1] Rashkin, Hannah. "Towards empathetic open-domain conversation models: A new benchmark and dataset." arXiv preprint arXiv:1811.00207 (2018).

[2] Sharma, Ashish, et al. "A computational approach to understanding empathy expressed in text-based mental health support." arXiv preprint arXiv:2009.08441 (2020).

[3] Welivita, Anuradha, and Pearl Pu. "A taxonomy of empathetic response intents in human social conversations." arXiv preprint arXiv:2012.04080 (2020).


# Acknowledgements

The following repository and database for the article EmpatheticExchanges: Towards Understanding the Cues for Empathy in Dyadic Conversations was developed with the help of the National Council of Humanities, Sciences, and Technology of Mexico (CONAHCYT),  Tecnologico de Monterrey and Honda Research Institute Japan Co., Ltd.

 We sincerely extend our gratitude to the following annotators for EmpatheticExchanges:

    • Dalia Janet Aguilera Álvarez
    • Ana Carla Díaz Aguirre
    • Maryali Heredia Alvarado
    • Zara Lucia Schevenin Pérez
    • Heidi Ailyn Bernaldez Acevedo

Likewise, we extend our gratitude to the following annotators for EmpatheticExchanges_extended

    • Alan Javier Castillo Moreno
    • Kimberly Melgar Sánchez
    • Diego Fernando Sabillon Chinchilla
    • París Abelardo Gómez Huerta
    • Rodrigo Andrey González Pérez
    • Andrés Felipe Gutiérrez Jaramillo
    • Leslie Denisse Muñoz Ruelas
    • Marco Antonio García Mendoza
    • Abraham Ureña Rodríguez
    • Ricardo González Franyutti
    • Erika Mariel Castillo López
    • Gilberto Valdez Arreola
    • Eduardo Daniel Montaño Ortega
    • Diego Velázquez Saldaña
    • David Alejandro González Ibarra
    • Alejandro Fernández del Valle Herrera
    • Luis Antonio Barajas Ramírez
    • Sergio Zuckerman Gittler
    • David Beltrán González
    • Leonardo Cossio Dinorín
    • Regina Pérez Caballero
    • Ana Karina Aramoni Ruíz
    • René Miguel Macías Olívar
    • Mariana García Villaseñor
    • Rodrigo Castellanos Rodríguez
    • Fernando Núñez Magallanes
    • Mariana León González
    • Óscar Ontiveros Ramírez
    • Jimena María Alanís Rubio

As well as other contributors who decided to remain anonymous.

# License

Please refer to the license file in the repository for information of the license.
