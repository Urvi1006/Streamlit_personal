from haystack.components.generators import HuggingFaceLocalGenerator
import torch
import pandas as pd


class LLM_Pipeline:

    def __init__(self):
        self.generator = HuggingFaceLocalGenerator("mistralai/Mistral-7B-Instruct-v0.1",
                                 huggingface_pipeline_kwargs={"device_map":"auto",
                                               "model_kwargs":{"load_in_4bit":True,
                                                "bnb_4bit_use_double_quant":True,
                                                "bnb_4bit_quant_type":"nf4",
                                                "bnb_4bit_compute_dtype":torch.bfloat16,
                                                },
                                                              },
                                 generation_kwargs={"max_new_tokens": 1500})
        # torch.cuda.set_device(1)

        self.prompt_initial_classifier = '''
    Act like an industry expert who wants to get rules from a given set of sentences. Your task will be to return the sentences that are rules as output.
A sentence can be called as rule if it has the following characteristics:
1. Rules can be converted into a meaningful mathematical expression.
  Here are some examples for your reference:
  a. Rule: The minimum distance from a countersink and a bend is four times the material thickness plus the bend radius.
     Expression : distance(countersink, bend) >= 4 * material.thickness + bend.radius
  b. Rule: The minimum radius of the bend should be at least 3 times the material thickness.
     Expression: Bend.Radius >= 3 * Material.Thickness
  c. Rule: The minimum radius of the fillet should be at least 0.125 inches.
     Expression: Fillet.Radius >= 0.125 in

2.The rules can be converted into an expression so they consist of two sides LHS and RHS.
  Here are some examples for your reference:
  a. Expression: distance(countersink, bend) >= 4 * material.thickness + bend.radius
     LHS : distance(countersink, bend)
     RHS : 4 * material.thickness + bend.radius
  b. Expression: Bend.Radius >= 3 * Material.Thickness
     LHS : Bend.Radius
     RHS : 3 * Material.Thickness
  c. Expression: Fillet.Radius >= 0.125 in
     LHS : Fillet.Radius
     RHS : 0.125 in

The rules specify quantifiable criteria that can be converted into mathematical expressions,
while the not-rules explain potential issues that may arise during manufacturing, but they do not specify quantifiable criteria.

For your reference I am giving you some examples of statements which are rules and not rules:
Rules:
1.	The minimum bend radius of a metal sheet should not be less than 1.5 times the material thickness.
2.	In welding applications, the weld penetration should be at least 50% of the base material thickness.
3.	For threaded connections, the engagement length of the threads should be at least 1.5 times the thread diameter.
4.	The minimum clearance between moving parts should be at least 0.1 times the sum of their diameters.
5.	The feature depth to corner radius ratio should not exceed 8.0.
6.	The outside radius of a curl should not be smaller than 2 times the material thickness.
7.	The cylindricity deviation of a cylindrical feature should be less than or equal to 0.02 mm (0.0008 in).
8.	In 3D printing, the minimum layer thickness should be greater than or equal to 0.1 mm.
9.	The draft angle for sidewalls should typically be between 0.5 to 2 degrees for inside and outside walls.
10.	The minimum distance between a cutout edge and a bend should be three times the material thickness plus the bend radius.

Not rules:
1. For the Additive Manufacturing process, if the gap between faces is too small, it may lead to difficulty in post-processing operations like cleaning, etc.
2. A draft is a slope or taper which is incorporated on sidewalls of a cast part.
3. Wall thickness variation should be within tolerance to allow for smooth filling of the mold.
4. The drill follows a path of least resistance when it intersects a cavity during machining.
5. Punching holes very near to each other could result in deformation of material between holes.
6. Weld-to-edge of part feature spacing should be adequate so that the electrodes can make proper contact with the joined surfaces without shunting to the adjacent wall.
7. Mold wall thickness will get affected due to spacing between various features in the plastic model.
8. It is not easy to generalize what the wall thickness of a part should be. The wall plays a part both in the design concept and embodiment.
9. If the distance from one Countersink hole to another Countersink hole is below a minimum value, then there is a chance of distortion/cracking of a sheet in the considered area.
10. If the distance from a dowel hole edge and cutout edge is below a minimum value, then there is a chance of distortion/cracking of sheet in the considered area.

Here is a paragraph from which you have to pick out sentences which are rules.
While giving output dont create sentences of your own and hallucinate only take sentences
from the paragraph.The rules specify quantifiable criteria that can be converted into mathematical expressions,
while the not-rules explain potential issues that may arise during manufacturing, but they do not specify quantifiable criteria.So do not print sentences which don't specify
quantifiable criteria. While printing out the rules crop out the part that is informative and doesn't contribute to the mathematical expression.
Here is the paragraph, pick out only rules from this paragraph dont print sentences that are not rules:
"
'''

        self.prompt_initial_ruletojson = '''
Convert into expression the statement. Also identify the objects present in the expression and express them in the json format

Following are the examples:

1. Statement: “The wall thicknesses should be less than 60 % of the nominal wall to minimize sinking”.

Converted Expression:
wall.thickness < 0.6 * nominal_wall.thickness

JSON format:

{
"expression": "wall.thickness < 0.6 * nominal_wall.thickness",
"operator": "<",
"objects": [
{
"object1": "Wall",
"attribute1": "thickness"
},
{
"object2": "nominal_wall",
"attribute2": "thickness"
}
]
}

2. Statement - “The thickness of the material should be at least 0.125 inches.”

Converted Expression:
material.thickness >= 0.125

JSON format:
{
"expression": "material.thickness >= 0.125",
"operator": ">=",
"objects": [
{
"object1": "Material",
"attribute1": "thickness"
}
]
}
3. Statement - “The minimum distance between two pins should be at least twice the diameter of the larger pin.”

Converted expression:
distance(pin1,pin2) >= 2 * max(pin1.diameter, pin2.diameter)

JSON format:
{
"expression": "distance(pin1,pin2) >= 2 * max(pin1.diameter, pin2.diameter)",
"operator": ">="
“Objects”: [
{
"object1": "pin",
"function": "distance",
"object2" : "pin"
},
{
"object3": "pin",
"attribute3": "diameter"
}
]
}

4. Statement: “The diameter of the cutter should be equal to the width of the key.”

Converted Expression:
cutter.diameter = key.width

JSON format:
{
"expression": "cutter.diameter = key.width",
"operator": "=",
"objects": [
{
"object1": "Cutter",
"attribute2": "diameter"
},
{
"object2": "Key",
"attribute2": "width"
}
]
}

5. Statement: “The minimum distance between countersinks should be at least 3 times the stock thickness.”

Converted Expression:
distance(countersink1, countersink2) >= 3 * stock.thickness

JSON format:
{
"expression": "distance(countersink1, countersink2) >= 3 * stock.thickness",
"operator": ">=",
"objects": [
{
"object1": "Countersink",
"function: "distance",
"object2" : "Countersink"
},
{
"object3": "Stock",
"attribute3": "thickness"
}
]
}

6. Statement : “In laser cutting, the cutting speed to material thickness ratio should be less than 1000.”

Converted Expression:
cutting_speed / material.thickness < 1000

JSON format:
{
"expression": "cutting_speed / material.thickness < 1000",
"operator": "<",
"objects": [
{
"object": "Laser Cutting",
"attribute": "cutting_speed"
},
{
"object": "Material",
"attribute": "thickness"
}
]
}


7. Statement: “For Boss Height to Local Wall Thickness ratio, the value should be less than or equal to 5.0.”

Converted Expression:
boss.height / local_wall.thickness <= 5.0

JSON format:
{
"expression": "boss.height / local_wall.thickness <= 5.0",
"operator": "<=",
"objects": [
{
"object1": "Boss",
"attribute1": "height"
},
{

"object2": "Wall",
"attribute2": "thickness"
}
]
}

8. Statement: “The dowel outer diameter should be greater than 3.0 mm and smaller than 5.0 mm.”

Converted expression:
3.0 < dowel.outer_diameter < 5.0

JSON format:

{
"expression": "3.0 < dowel.outer_diameter < 5.0",
"operator": "<",
"objects": [
{
"object": "Dowel",
"attribute": "outer_diameter"
}
]
}

9. Statement : “The minimum distance from bend to parallel slot should be at least 4 times sheet thickness”

Converted Expression:
distance(bend,parallel_slot)  >= 4 * sheet.thickness

JSON format:
{
"expression": "distance(bend,parallel_slot)" >= 4 * sheet_thickness",
"operator": ">=",
"objects": [
{
"object1": "Bend",
"function" : "distance",
"object2": "parallel_slot"
},
{
"object3": "Sheet",
"attribute3": "thickness"
}
]
}
10. Outside radius must be at least twice the material's thickness.

Converted Expression:
radius(outside) >= 2 * material.thickness

JSON format:
{
"expression": "radius(outside) >= 2 * material.thickness",
"operator": ">=",
"objects": [
{
"object1": "Radius",
},
{
"object2": "Material",
"attribute2": "thickness"
}
]
}

11. Keep hole and slot diameters at least as large as material thickness. Higher strength materials require larger diameters.

Converted Expression:
Hole.diameter >= material.thickness
Slot.diameter >= material.thickness

JSON format:
{
"expression": "Hole.diameter >= material.thickness",
"operator": ">=",
"objects": [
{
"object1": "Hole",
"attribute1": "diameter"
},
{
"object2": "Material",
"attribute2": "thickness"
}
]
}
JSON format:
{
"expression": "Slot.diameter >= material.thickness",
"operator": ">=",
"objects": [
{
"object1": "Slot",
"attribute1": "diameter"
},
{
"object2": "Material",
"attribute2": "thickness"
}
]
}

12.Tabs must be at least 0.126" (3.2mm) thick, or two times the material's thickness, whichever is greater; the length must also be no larger than 5 times its width.

Converted Expression:
Tab.thickness >= max(0.126, 2 * material.thickness)
Tab.length <= 5 * tab.width
JSON format:
{
"expression": "Tab.thickness >= max(0.126, 2 * material.thickness)",
"operator": ">=",
"objects": [
{
"object1": "Tab",
"attribute1": "thickness"
},
{
"object2": "Material",
"attribute2": "thickness"
}
]
}
JSON format:
{
"expression": "Tab.length <= 5 * tab.width",
"operator": ">=",
"objects": [
{
"object1": "Tab",
"attribute1": "length"
},
{
"object2": "Tab",
"attribute2": "width"
}
]
}

13. Notches and tabs should not be narrower than 1.5 times the material thickness. 

Converted Expression:
Notch.width >= 1.5 * material.thickness
Tab.width >= 1.5 * material.thickness

JSON format:
{
"expression": "Notch.width >= 1.5 * material.thickness",
"operator": ">=",
"objects": [
{
"object1": "Notch",
"attribute1": "width"
},
{
"object2": "Material",
"attribute2": "thickness"
}
]
}

JSON format:
{
"expression": "Tab.width >= 1.5 * material.thickness",
"operator": ">=",
"objects": [
{
"object1": "Tab",
"attribute1": "width"
},
{
"object2": "Material",
"attribute2": "thickness"
}
]
}

14. Open hems: Minimum inside diameter must be at least 1X MT (Here consider MT as material thickness)

Converted expression: Open_hem.diameter >= 1 * material.thickness

JSON format:
{
"expression": "Open_hem.diameter >= 1 * material.thickness",
"operator": ">=",
"objects": [
{
"object1": "Open_hem",
"attribute1": "diameter"
},
{
"object2": "Material",
"attribute2": "thickness"
}
]
}

15. The minimum distance that a gusset should be from the edge of a hole in a parallel plane is eight times the material thickness plus the radius of the gusset.

Converted expression: distance(gusset, hole_edge) >= 8 * material.thickness + gusset.radius

JSON format:
{
"expression": "distance(gusset, hole_edge) >= 8 * material.thickness + gusset.radius",
"operator": ">=",
"objects": [
{
"object1": "gusset",
"function": "distance"
"object2": "hole_edge"
},
{
"object3": "Material",
"attribute3": "thickness"
},
{
"object4": "Gusset",
"attribute4": "radius"
}
]
}

Now provide output for the below statement

Statement - '''

    def json_from_string(self, rule: str) -> str:                                          #generates rule for a single sentence
        self.generator.warm_up()
        prompt = self.prompt_initial_ruletojson + rule
        result = self.generator.run(prompt)["replies"][0]

        return result

    def json_from_list(self, rules: list, csv_path: str):                                  #generates rules for a list of rules and saves a csv in the same location as the source pdf
        self.generator.warm_up()
        final_df = pd.DataFrame(columns = ['Rules', 'Expression'], dtype="str") #Emtpy DataFrame created with columns Rules and Expression
        for i in range(0, len(rules)):
            rule = str(rules[i])
            prompt = self.prompt_initial_ruletojson + rule
            result = self.generator.run(prompt)["replies"][0]
            final_df.loc[final_df.shape[0]] = [rules[i], result]                #Adds a new row to the DataFrame

        final_df.to_csv(csv_path)
       

    def classifier(self, pdf_text_data: list[str]) -> list:                                #classifies the data into rules and not rules
        self.generator.warm_up()
        rules_string = ""

        process_line = ""
        for line in pdf_text_data:
            
                           #Set the maximum possible token length to near 4000. Limit of Mistral is 7000
            process_line += line

            if len(process_line.split()) < 1000:
                continue    
                 
            prompt = self.prompt_initial_classifier + process_line + '"'
            result = self.generator.run(prompt)["replies"][0]
            rules_string += result                           #Splits the generated sentences into list of strings at [ & ] tokens
            process_line = ""

        rules_list = rules_string.splitlines()
        return [i for i in rules_list if i]

    def pdf_query_function(self, pdf_text_data: list, pdf_path: str):                 #Function which takes in the pdf_text_data and the pdf path for mass processing of relevant data
        self.json_from_list(self.classifier(pdf_text_data), self.csv_path_gen(pdf_path))         #Takes the returned list of rules from classifier to the from_list funcntion and generates a .csv file

    def csv_path_gen(self, pdf_path: str):                                            #Generates the .csv file path in the same location as the source pdf
        csv_path = pdf_path[:-4]
        return csv_path + "_result.csv"


       