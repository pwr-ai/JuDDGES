from enum import Enum

from pydantic import BaseModel as PydanticModel
from pydantic import Field

from label_studio_toolkit.schemas.utils import SchemaUtilsMixin


class BaseModel(PydanticModel):
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict):
            for field_name, field in cls.__annotations__.items():
                if field_name in obj and obj[field_name] is not None:
                    origin = getattr(field, "__origin__", None)
                    if origin is list and isinstance(obj[field_name], str):
                        obj[field_name] = [obj[field_name]]

        return super().model_validate(obj, *args, **kwargs)


class ConfessPleadGuiltyEnum(str, Enum):
    YES = "Yes"
    NO = "No"


class RemandDecisionEnum(str, Enum):
    REMANDED_INTO_CUSTODY = "Remanded into custody"
    UNCONDITIONAL_BAIL = "Unconditional Bail"
    CONDITIONAL_BAIL = "Conditional Bail"
    NO = "No"


class SentServeEnum(str, Enum):
    CONCURRENT = "Concurrent"
    CONCURRENTLY = "Concurrently"
    COMBINATION = "Combination"
    CONSECUTIVE = "Consecutive"
    CONSECUTIVELY = "Consecutively"
    SINGLE = "Single"


class OffSexEnum(str, Enum):
    ALL_FEMALE = "All Female"
    ALL_MALE = "All Male"
    MIXED = "Mixed"


class OffJobOffenceEnum(str, Enum):
    EMPLOYED = "Employed"
    UNEMPLOYED = "Unemployed"
    STUDENT = "Student"
    CHILD = "Child"
    RETIRED = "Retired"
    OTHER = "Other"


class OffHomeOffenceEnum(str, Enum):
    FIXED_ADDRESS = "Fixed Address"
    TEMPORARY_ACCOMMODATION = "Temporary Accommodation"
    HOMELESS = "Homeless"
    SECTIONED = "Sectioned"


class OffMentalOffenceEnum(str, Enum):
    HAD_MENTAL_HEALTH_PROBLEMS = "Had mental health problems"
    LEARNING_DEVELOPMENTAL = "Learning/developmental"
    HAS_LEARNING_DIFFICULTIES = "Has learning difficulties"
    OTHER = "Other"


class OffIntoxOffenceEnum(str, Enum):
    YES_DRINKING = "Yes-drinking"
    YES_DRINKING_DRUGS = "Yes-drinking&drugs"
    YES_DRUGS = "Yes-drugs"
    NO = "No"


class OffVicRelationEnum(str, Enum):
    ACQUAINTANCE = "Acquaintance"
    STRANGER = "Stranger"
    RELATIVE = "Relative"


class VictimTypeEnum(str, Enum):
    INDIVIDUAL_PERSON = "Individual person"
    INDIVIDUALS = "Individuals"
    COMPANY = "Company"
    ORGANISATION = "Organisation"


class VicSexEnum(str, Enum):
    ALL_FEMALE = "All Female"
    ALL_MALE = "All Male"
    MIXED = "Mixed"


class VicJobOffenceEnum(str, Enum):
    EMPLOYED = "Employed"
    UNEMPLOYED = "Unemployed"
    STUDENT = "Student"
    RETIRED = "Retired"
    CHILD = "Child"
    OTHER = "Other"


class VicHomeOffenceEnum(str, Enum):
    HOMELESS = "Homeless"
    FIXED_ADDRESS = "Fixed Address"
    TEMPORARY_ACCOMMODATION = "Temporary Accommodation"
    SECTIONED = "Sectioned"


class VicMentalOffenceEnum(str, Enum):
    HAD_MENTAL_HEALTH_PROBLEMS = "Had mental health problems"
    LEARNING_DEVELOPMENTAL = "Learning/developmental"
    HAS_LEARNING_DIFFICULTIES = "Has learning difficulties"
    OTHER = "Other"


class VicIntoxOffenceEnum(str, Enum):
    YES_DRINKING = "Yes-drinking"
    YES_DRINKING_DRUGS = "Yes-drinking&drugs"
    YES_DRUGS = "Yes-drugs"
    NO = "No"


class PreSentReportEnum(str, Enum):
    HIGH_RISK_HARM = "High risk of harm"
    LOW_RISK_REOFFENDING = "Low risk of reoffending"
    HIGH_RISK_REOFFENDING = "High risk of reoffending"
    MEDIUM_RISK_REOFFENDING = "Medium risk of reoffending"
    MEDIUM_RISK_HARM = "Medium risk of harm"
    LOW_RISK_HARM = "Low risk of harm"


class VicImpactStatementEnum(str, Enum):
    YES = "Yes"
    NO = "No"


class AppellantEnum(str, Enum):
    OFFENDER = "Offender"
    ATTORNEY_GENERAL = "Attorney General"
    APPELLANT = "Appellant"
    OTHER = "Other"


class AppealCourtAnnotation(BaseModel, SchemaUtilsMixin):
    ConvCourtName: list[str] | None = Field(
        None,
        description="Name(s) of the court where the defendant was convicted or pleaded guilty. Example: ['Crown Court at Southwark']",
    )
    ConvictPleaDate: list[str] | None = Field(
        None,
        description="Date(s) on which the defendant was convicted or pleaded guilty. Example: ['2003-01-22']",
    )
    ConvictOffence: list[str] | None = Field(
        None,
        description="Offence(s) of which the defendant was convicted. Example: ['Dangerous driving (death by)', 'Aggravated burglary', 'Supply of controlled drugs (including conspiracy to)', 'Manslaughter', 'Pervert the course of justice', 'Intent to endanger life']",
    )
    AcquitOffence: list[str] | None = Field(
        None,
        description="Offence(s) of which the defendant was acquitted. Example: ['Theft', 'Money laundering', 'Criminal damage', 'Perjury']",
    )
    ConfessPleadGuilty: list[ConfessPleadGuiltyEnum] | None = Field(
        None, description="Did the defendant confess or plead guilty? Example: ['Yes']"
    )
    PleaPoint: list[str] | None = Field(
        None,
        description="Stage at which the plea was entered. Example: ['in police presence', 'at first court appearance', 'on day of trial', 'on re-arraignment', 'earliest opportunity']",
    )
    RemandDecision: list[RemandDecisionEnum] | None = Field(
        None, description="Remand decision post-conviction. Example: ['Unconditional Bail']"
    )
    RemandCustodyTime: list[int] | None = Field(
        None, description="Duration in days of any remand in custody. Example: [4]"
    )
    SentCourtName: list[str] | None = Field(
        None,
        description="Name(s) of the court where the defendant was sentenced. Example: ['Crown Court at Canterbury', 'Newcastle Crown Court', 'Southend Crown Court']",
    )
    Sentence: list[str] | None = Field(
        None, description="Sentence(s) imposed. Example: ['2 years imprisonment', '£500 fine']"
    )
    SentServe: list[SentServeEnum] | None = Field(
        None, description="How sentences run. Example: ['Concurrent']"
    )
    WhatAncillary: list[str] | None = Field(
        None,
        description="Ancillary orders applied by the court. Example: ['Restraining order', 'Compensation order']",
    )
    OffSex: list[OffSexEnum] | None = Field(
        None, description="Gender(s) of the defendant(s). Example: ['All Female']"
    )
    OffAgeOffence: list[int] | None = Field(
        None, description="Age of defendant at offence. Example: [21, 34]"
    )
    OffJobOffence: list[OffJobOffenceEnum] | None = Field(
        None, description="Employment status at offence. Example: ['Employed']"
    )
    OffHomeOffence: list[OffHomeOffenceEnum] | None = Field(
        None, description="Accommodation status at offence. Example: ['Fixed Address']"
    )
    OffMentalOffence: list[OffMentalOffenceEnum] | None = Field(
        None,
        description="Learning/developmental or mental-health issues noted. Example: ['Had mental health problems']",
    )
    OffIntoxOffence: list[OffIntoxOffenceEnum] | None = Field(
        None, description="Intoxication status. Example: ['Yes-drugs']"
    )
    OffVicRelation: list[OffVicRelationEnum] | None = Field(
        None, description="Relationship defendant→victim. Example: ['Acquaintance']"
    )
    VictimType: list[VictimTypeEnum] | None = Field(
        None, description="Type of victim. Example: ['Individual person']"
    )
    VicNum: list[str] | None = Field(
        None, description="Number of victims or ratio. Example: ['two of 36']"
    )
    VicSex: list[VicSexEnum] | None = Field(
        None, description="Gender(s) of victim(s). Example: ['All Female']"
    )
    VicAgeOffence: list[int] | None = Field(
        None, description="Age of victim(s) at offence. Example: [31]"
    )
    VicJobOffence: list[VicJobOffenceEnum] | None = Field(
        None, description="Employment status of victim(s). Example: ['Employed']"
    )
    VicHomeOffence: list[VicHomeOffenceEnum] | None = Field(
        None, description="Accommodation status of victim(s). Example: ['Homeless']"
    )
    VicMentalOffence: list[VicMentalOffenceEnum] | None = Field(
        None,
        description="Learning/developmental or mental-health issues for victim(s). Example: ['Had mental health problems']",
    )
    VicIntoxOffence: list[VicIntoxOffenceEnum] | None = Field(
        None, description="Victim's intoxication status. Example: ['Yes-drugs']"
    )
    ProsEvidTypeTrial: list[str] | None = Field(
        None,
        description="Evidence types by prosecution. Example: ['CCTV', 'DNA match', 'Victim testimony', 'Expert report/testimony']",
    )
    DefEvidTypeTrial: list[str] | None = Field(
        None,
        description="Evidence types by defence. Example: ['Offender denies offence', 'No DNA evidence', 'Alibi claim']",
    )
    PreSentReport: list[PreSentReportEnum] | None = Field(
        None, description="Risk level from pre-sentence report. Example: ['High risk of harm']"
    )
    AggFactSent: list[str] | None = Field(
        None,
        description="Aggravating factors at sentencing. Example: ['offence committed while on bail', 'use of a weapon to frighten or injure victim']",
    )
    MitFactSent: list[str] | None = Field(
        None,
        description="Mitigating factors at sentencing. Example: ['offender showed genuine remorse', 'Offender has no relevant previous convictions']",
    )
    VicImpactStatement: list[VicImpactStatementEnum] | None = Field(
        None, description="Was a victim impact statement provided? Example: ['Yes']"
    )
    Appellant: list[AppellantEnum] | None = Field(
        None, description="Who brings the appeal. Example: ['Offender']"
    )
    CoDefAccNum: list[int] | None = Field(
        None, description="Number of co-defendants/co-accused. Example: [2]"
    )
    AppealAgainst: list[str] | None = Field(
        None,
        description="Ground(s) for appeal. Example: ['Conviction is unsafe', 'Sentence is unduly lenient', 'Conviction unsafe', 'appeal against sentence']",
    )
    AppealGround: list[str] | None = Field(
        None,
        description="Specific legal grounds of appeal. Example: ['jury exposed to prejudicial evidence', 'trial judge misdirected jury', 'excessive sentence', 'co-def received 33% credit/reduction in sentence, appellant received only 25%']",
    )
    SentGuideWhich: list[str] | None = Field(
        None,
        description="Sentencing guidelines or statutes cited such as s./section __ of (the) __ Act __ or [Sentencing Council's (definitive) Guideline/s on/for… /Totality (principle/ of)…. Example: ['section 25(1)(a) of the Identity Cards Act 2006', 'section 155 of the Powers of Criminal Courts (Sentencing) Act 2000', 'section 52 of the Firearms Act 1968']",
    )
    AppealOutcome: list[str] | None = Field(
        None,
        description="Outcome of the appeal. Example: ['Dismissed', 'Allowed & Conviction Quashed', 'Dismissed-Failed', 'Appeal allowed and sentence reduced by 106 days']",
    )
    ReasonQuashConv: list[str] | None = Field(
        None,
        description="Reasons for quashing conviction. Example: ['The confiscation orders could not be upheld', 'indictment charged the appellant in an impermissible manner', 'procedural irregularity', 'jury misdirection']",
    )
    ReasonSentExcessNotLenient: list[str] | None = Field(
        None,
        description="Reasons why sentence was unduly excessive. Example: ['lack of consideration for personal mitigation', 'Mitigating factors should have been considered, such as young age of offenders', 'seriousness of offence is too high']",
    )
    ReasonSentLenientNotExcess: list[str] | None = Field(
        None,
        description="Reasons why sentence was unduly lenient. Example: ['judge did not refer to relevant guideline', 'Offence difficult to sentence as it happened in the past when offender was a child']",
    )
    ReasonDismiss: list[str] | None = Field(
        None,
        description="Reasons for dismissal of the appeal. Example: ['original sentence fell within guideline range', 'no ground for appeal', 'judge made correct ruling', 'no prejudice caused to offender', 'judge categorised offence correctly']",
    )
