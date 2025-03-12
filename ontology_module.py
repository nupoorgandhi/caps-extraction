from pydantic import BaseModel
from typing import Optional

class SpanCluster(BaseModel):
    label: str
    label_description: str
    spans: list[str]

class Clusters(BaseModel):
    clusters: list[SpanCluster]

class Mechanism(BaseModel):
    education_outreach: bool
    mandate: bool
    incentive: bool
    disincentive: bool
    voluntary_action: bool
    tax: bool
    regulation: bool
    fee: bool
    
    objective: list[str]
    compliance: list[str]
    action_description: list[str]
    authority: list[str]
    target: list[str]

class Duration(BaseModel):
    start_date: str
    end_date: str
    length_of_time: str
    elapsed: bool
    in_progress: bool
    still_needs_to_be_initiated: bool

class ClimateHazard(BaseModel):
    primary_impact: list[str]
    secondary_impact: list[str]
    tertiary_human_impact: list[str]
    exposure_subpopulation: list[str]
    exposure_geographic_area: list[str]
    severity_area: list[str]
    duration: Duration

class GHGEmission(BaseModel):
  emission_source: list[str]
  emission_type: list[str]
  quantity: str
  unit: str
  duration: Duration
  emitter_stakeholder: list[str]  
  affected_by_emission_stakeholder: list[str]

class EmissionsReductionPolicy(BaseModel):
    emission: list[GHGEmission]
class AdaptationPolicy(BaseModel):
    climate_hazards: list[ClimateHazard]

class SupportingSpan(BaseModel):
    supporting_evidence_spans: list[str]

class Implementation(BaseModel):
    duration: Duration
    funding: list[str]
    resources_necessary: list[str]
    resources_allocated: list[str]
    evaluation_criteria: list[str]
    feedback_mechanism: list[str]
    
# class Policy(BaseModel):
#     is_adaptation_policy: bool
#     is_emissions_reduction_policy: bool
#     policy_name: str
#     representation_if_adaptation_policy: Optional[AdaptationPolicy]
#     representation_if_emissions_reduction_policy: Optional[EmissionsReductionPolicy]
    
#     policy_action_mechanisms: list[Mechanism]
#     implementation_features: list[Implementation]
#     co_benefits: list[str]
#     qualifying_phrase: list[str]
#     deontic: list[str]
  
class Policy(BaseModel):
  policy_description: str
  is_adaptation_policy: bool
  is_emissions_reduction_policy: bool
  sector: str
  target_or_performance_metric: str
  ghg_reduction_2020: str
  ghg_reduction_2025: str
  ghg_reduction_2030: str
  ghg_reduction_2035: str
  ghg_reduction_2040: str
  ghg_reduction_2050: str
  responsibility_or_implementation_details: str
  responsibility_or_implementation: str
  already_projected_amount: str
  quantification_of_ghg_emissions_reductions: str
  relative_cost: str
  costs_and_benefits_private: str
  costs_and_benefits_public: str
  funding: str
  timeframe: str
  co_benefits: str
  supporting_activities: str
  target_year: str



class PolicyList(BaseModel):
  policy_list: list[Policy]




SCHEMA = {
  "response_format":"json_schema",
  "name": "policy_collection",
  "schema": {
    "type": "object",
    "properties": {
      "policy_list": {
        "type": "array",
        "description": "A collection of policies.",
        "items": {
          "type": "object",
          "properties": {
            "is_adaptation_policy": {
              "type": "boolean",
              "description": "Indicates if the policy is an adaptation policy."
            },
            "is_emissions_reduction_policy": {
              "type": "boolean",
              "description": "Indicates if the policy is focused on emissions reduction."
            },
            "policy_name": {
              "type": "string",
              "description": "The name of the policy."
            },
            "representation_if_adaptation_policy": {
              "type": "object",
              "properties": {
                "climate_hazards": {
                  "type": "array",
                  "description": "A list of climate hazards addressed by the adaptation policy.",
                  "items": {
                    "$ref": "#/$defs/climate_hazard"
                  }
                }
              },
              
              "additionalProperties": False
            },
            "representation_if_emissions_reduction_policy": {
              "type": "object",
              "properties": {
                "emission": {
                  "type": "array",
                  "description": "A list of greenhouse gas emissions associated with the policy.",
                  "items": {
                    "$ref": "#/$defs/ghg_emission"
                  }
                }
              },
              
              "additionalProperties": False
            },
            "policy_action_mechanisms": {
              "type": "array",
              "description": "Mechanisms for action",
              "items": {
                "$ref": "#/$defs/mechanism"
              }
            },
            "implementation_features": {
              "type": "array",
              "description": "Implementation features of the policy.",
              "items": {
                "$ref": "#/$defs/implementation"
              }
            },
            "co_benefit": {
              "type": "array",
              "description": "Co-benefits of the policy.",
              "items": {
                "type": "string"
              }
            },
            "qualifying_phrase": {
              "type": "array",
              "description": "Qualifying phrases related to the policy.",
              "items": {
                "type": "string"
              }
            },
            "deontic": {
              "type": "array",
              "description": "Deontic statements related to the policy.",
              "items": {
                "type": "string"
              }
            }
          },
          
          "additionalProperties": False
        }
      }
    },
    "required": [
      "policy_list"
    ],
    "additionalProperties": False,
    "$defs": {
      "mechanism": {
        "type": "object",
        "properties": {
          "education_outreach": {
            "type": "boolean"
          },
          "mandate": {
            "type": "boolean"
          },
          "incentive": {
            "type": "boolean"
          },
          "disincentive": {
            "type": "boolean"
          },
          "voluntary_action": {
            "type": "boolean"
          },
          "tax": {
            "type": "boolean"
          },
          "regulation": {
            "type": "boolean"
          },
          "fee": {
            "type": "boolean"
          },
          "objective": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "compliance": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "action_description": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "authority": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "target": {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        },
        
        "additionalProperties": False
      },
      "duration": {
        "type": "object",
        "properties": {
          "start_date": {
            "type": "string"
          },
          "end_date": {
            "type": "string"
          },
          "length_of_time": {
            "type": "string"
          },
          "elapsed": {
            "type": "boolean"
          },
          "in_progress": {
            "type": "boolean"
          },
          "still_needs_to_be_initiated": {
            "type": "boolean"
          }
        },
        
        "additionalProperties": False
      },
      "climate_hazard": {
        "type": "object",
        "properties": {
          "primary_impact": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "secondary_impact": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "tertiary_human_impact": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "exposure_subpopulation": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "exposure_geographic_area": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "severity_area": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "duration": {
            "$ref": "#/$defs/duration"
          }
        },
        
        "additionalProperties": False
      },
      "ghg_emission": {
        "type": "object",
        "properties": {
          "emission_source": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "emission_type": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "quantity": {
            "type": "string"
          },
          "unit": {
            "type": "string"
          },
          "duration": {
            "$ref": "#/$defs/duration"
          },
          "emitter_stakeholder": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "affected_by_emission_stakeholder": {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        },
        
        "additionalProperties": False
      },
      "implementation": {
        "type": "object",
        "properties": {
          "duration": {
            "$ref": "#/$defs/duration"
          },
          "funding": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "resources_necessary": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "resources_allocated": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "evaluation_criteria": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "feedback_mechanism": {
            "type": "array",
            "items": {
              "type": "string"
            }
          }
        },
        
        "additionalProperties": False
      }
    }
  },
  "strict": True
}