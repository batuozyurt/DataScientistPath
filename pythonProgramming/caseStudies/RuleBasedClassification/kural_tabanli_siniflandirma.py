#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
#region Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.
#endregion

#############################################
# Veri Seti Hikayesi
#############################################
#region Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı
#endregion

#############################################
import pandas as pd

df = pd.read_csv("persona.csv")
df.shape
df.info
df.head()

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

df["PRICE"].nunique()

df["COUNTRY"].value_counts()
df.groupby("COUNTRY").agg({"PRICE": "count"})
df.pivot_table(values="PRICE", index="COUNTRY", aggfunc="count")

df.groupby("COUNTRY")["PRICE"].sum()
df.groupby("COUNTRY").agg({"PRICE": "sum"})
df.pivot_table(values="PRICE", index="COUNTRY", aggfunc="sum")

df.groupby("SOURCE").agg({"PRICE": "count"})
df.pivot_table(values="PRICE", index="SOURCE", aggfunc="count")

df.groupby("COUNTRY").agg({"PRICE": "mean"})
df.pivot_table(values="PRICE", index="COUNTRY", aggfunc="mean")

df.groupby("SOURCE").agg({"PRICE": "mean"})
df.pivot_table(values="PRICE", index="SOURCE", aggfunc="mean")

df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})
df.pivot_table(values="PRICE", index=["COUNTRY", "SOURCE"], aggfunc="mean")

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
list = ["COUNTRY", "SOURCE", "SEX", "AGE"]
df.groupby(list).agg({"PRICE": "mean"})
df.pivot_table(values="PRICE", index=list, aggfunc="mean")

agg_df = df.groupby(list).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()

agg_df.reset_index(inplace=True)
agg_df.head()

limits = [0, 18, 23, 30, 40, agg_df["AGE"].max()] # age değişkenindeki max değerle sonlandırıyorum.
dotNames = ["0_18", "19_23", "24_30", "31_40", "41_" + str(agg_df["AGE"].max())]
agg_df["age_categorical"] = pd.cut(agg_df["AGE"], limits, labels=dotNames)
agg_df.head()

agg_df["customers_level_based"] = ["_" + _.upper() for _ in agg_df.drop(["AGE", "PRICE"], axis=1).values]
agg_df.head()
agg_df["customers_level_based"].value_counts()
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df.head()
agg_df.reset_index(inplace=True)
agg_df["customers_level_based"].value_counts()

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head()
agg_df.groupby("SEGMENT").agg({"PRICE": ["count", "max", "min", "mean", "sum"]})

#############################################

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
