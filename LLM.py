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
                                 generation_kwargs={"max_new_tokens": 400})
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
       

    def classifier(self, pdf_text_data: list) -> list:                                #classifies the data into rules and not rules
        self.generator.warm_up()
        rules_list = []

        for i in range(0, len(pdf_text_data)):
            process_line = ""
            while(len(process_line.split()) < 4000):                             #Set the maximum possible token length to near 4000. Limit of Mistral is 7000
                process_line += pdf_text_data[i]
                i += 1
            
            prompt = self.prompt_initial_classifier + process_line + '"'
            result = self.generator.run(prompt)["replies"][0]
            rules_list.append(result.split("[]"))                           #Splits the generated sentences into list of strings at [ & ] tokens
            
        print(rules_list)
        return rules_list

    def pdf_query_function(self, pdf_text_data: list, pdf_path: str):                 #Function which takes in the pdf_text_data and the pdf path for mass processing of relevant data
        self.from_list(self.classifier(pdf_text_data), self.csv_path_gen(pdf_path))         #Takes the returned list of rules from classifier to the from_list funcntion and generates a .csv file

    def csv_path_gen(self, pdf_path: str):                                            #Generates the .csv file path in the same location as the source pdf
        csv_path = pdf_path[:-4]
        return csv_path + "_result.csv"


if __name__ == "__main__":
    abc = LLM_Pipeline()
    abc.classifier(
        '''
1INJECTION MOLDINGDESIGN GUIDELINES ­­­­INJECTION MOLDED PARTSInjection molding is used for manufacturing a wide variety of parts, from small components like AAA battery boxes to large components like truck body panels.

Once a component is designed, a mold is made and preci-sion machined to form the features of the desired part.

The injection mold-ing takes place when a thermoplastic or thermoset plastic material is fed into a heated barrel, mixed, and forced into the metal mold cavity where it cools and hardens before being removed.

TOOLINGMold and die are used interchangeably to describe the tooling applied to produce plastic parts.

They are typically constructed from pre-hardened steel, hardened steel, aluminum, and/or beryllium-copper alloy.

Of these materials, hardened steel molds are the most expensive to make, but offer the user a long lifespan, which offsets the cost per part by spreading it over a larger quantity.

For low volumes or large components, pre-hardened steel molds provide a less wear-resistant and less expensive option.

The most economical molds are produced out of aluminum.

When de-signed and built using CNC machines or Electrical Discharge Machining processes, these molds can economically produce tens of thousands to hundreds of thousands of parts.

Note that beryllium copper is often used in areas of the mold that require fast heat removal or places that see the most shear heat generated.

INJECTION MOLDINGThe injection molding process uses a granular plastic that is gravity fed from a hopper.

A screw-type plunger forces the material into a heated chamber, called a barrel, where it is melted.

The plunger continues to advance, push-ing the polymer through a nozzle at the end of the barrel that is pressed against the mold.

The plastic enters the mold cavity through a gate and runner system.

After the cavity is filled, a holding pressure is maintained to compensate for material shrinkage as it cools.

At this same time, the

STRATASYSDIRECT.COM | 888-311-10172­­­WALL SECTION CONSIDERATIONSWALL THICKNESSCost savings are highest when components have a mini-mum wall thickness, as long as that thickness is con-sistent with the part’s function and meets all mold filling considerations.

As would be expected, parts cool faster with thin wall thicknesses, which means that cycle times are shorter, resulting in more parts per hour.

Further, thin parts weigh less, using less plastic per part.

On average, the wall thickness of an injection molded part ranges from 2mm to 4mm (.080 inch to .160 inch).

Thin wall injection molding can produce walls as thin as .05mm (.020 inch).

UNIFORM WALLSParts with walls of uniform thickness allow the mold cavity to fill more easily since the molten plastic does not have to be forced through varying restrictions as it fills.

If the walls are not uniform the thin section cools first, then as the thick section cools and shrinks it builds stresses near the boundary area between the two.

Be-cause the thin section has already hardened, it doesn’t yield.

As the thick section yields, it leads to warping or twisting of the part, which, if severe enough, can cause cracks.

Figure 1: Uniform wall thickness can reduce or eliminate warpingINJECTION MOLDING MATERIALSMaterials Selection: Many types of thermoplastic ma­terials are available.

Selection depends on the specific application.

The chart below shows some of the most common materials being used.

INJECTION MOLDING ENGINEERED THERMOPLASTIC MATERIALSNylonsPolyphenylene Sulfide PPSPolycarbonatesPolyehter SulfoneAcetalsPolyetheretherketone PEEKAcrylicsFluoropolymersPolypropylenesPolyether Imide PEIPolyethylenesPolyphenylene Oxide PPOAcrylonitrile Butadiene StyrenePolyurethanes PURThermoplastic ElastomersPolyphthalamide PPAscrew turns so that the next shot is moved into a ready position, and the barrel retracts as the next shot is heated.

Because the mold is kept cold, the plastic solidifies soon after the mold is filled.

Once the part inside the mold cools completely, the mold opens, and the part is ejected.

The next injection molding cycle starts the moment the mold closes and the polymer is injected into the mold cavity.

3VOIDS AND SHRINKAGETroublesome shrinkage problems can be caused by the intersection of walls that are not uniform in wall thick­ness.

Examples might include ribs, bosses, or any other projection of the nominal wall.

Since thicker walls solidify slower, the area they are attached to at the nominal wall will shrink as the projection shrinks.

This can result in a sunken area in the nominal wall.

Such shrinkage can be minimized if a rib thickness is maintained to between 50 and 60 percent of the walls they are attached to.

To further our example, bosses located into a corner will produce very thick walls, causing sink, unless isolated as in the illustration below.

Figure 5: Boss design to eliminate sinksWARPAGEThe dynamic of thin and thick sections and their cooling times creates warping as well.

As would be expected, as a thick section cools it shrinks, and the material for the shrinkage comes from the unsolidified areas causing the part to warp.

Other causes for warping might include the molding pro­cess conditions, injection pressures, cooling rates, packing problems, and mold temperatures.

Resin manufacturers’ process guidelines should be followed for best results.

Figure 6: Warpage caused by non-uniform wall thicknessWhat if you cannot have uniform walls (due to design limitations)?

If design limitations make it impossible to have uniform wall thicknesses, the change in thickness should be as gradual as possible.

Coring is a method where plastic is removed from the thick area, which helps to keep wall sections uniform, eliminating the problem altogether.

Figure 2: Transition of wall thicknessGussets are support structures that can be designed into the part to reduce the possibility of warping.

Figure 3: Coring to eliminate sinksFigure 4: Gusseting to reduce warping

STRATASYSDIRECT.COM | 888-311-10174RIBS Ribs are used in a design to increase the bending stiff­ness of a part without adding thickness.

Ribs increase the moment of inertia, which increases the bending stiffness.

Bending Stiffness = E (young’s Modulus) x I (Moment of Inertia)Rib thickness should be less than wall thickness to min­imize sinking effects.

The recommended rib thickness should not exceed 60 percent of the nominal thickness.

Plus, the rib should be attached with corner radii as generous as possible.

Figure 9: Proper rib design reduces sinkingFigure 7: Boss design guidelinesWall thicknesses for bosses should be less than 60 per­cent of the nominal wall to minimize sinking.

However, if the boss is not in a visible area, then the wall thick­ness can be increased to allow for increased stresses imposed by self-tapping screws.

BOSSESBosses are used to facilitate the registration of mating parts, for attaching fasteners such as screws, or for ac­cepting threaded inserts.

Figure 8: Boss strengthening techniqueThe base radius should be a minimum of 0.25 X thick­ness.

Bosses can be strengthened by incorporating gus­sets at the base or by using connecting ribs attaching to nearby walls.

5RIB INTERSECTIONSBecause the thickness of the material will be greater at the rib intersections, coring or another means of ma­terial removal should be employed to avoid excessive sinking from occurring on the opposite side.

Figure 10: Coring at rib intersectionsRIB GUILDELINESThe height of a rib should be limited to less than three times its thickness.

It is better to use multiple ribs to increase bending stiffness than to use one very tall rib.

RIB/LOAD AFFECT ON STIFFNESSA rib is oriented in such a way as to provide maximum bending stiffness to the part.

By paying attention to part geometry, designers must be conscious of the orienta­tion of the rib to the bending load or there will be no increase in stiffness.

Figure 11: Design guidelines for ribsFigure 12: Rib/ load orientation affects part stiffness; Draft angles for ribs should be a minimum of 0.25 to 0.5 degree of draft per sideDRAFT AND TEXTUREMold drafts facilitate part removal from the mold.

The draft must be in an offset angle that is parallel to the mold opening and closing.

The ideal draft angle for a given part depends on the depth of the part in the mold and its required end-use function.

Figure 12: Draft AngleAllowing for as much draft as possible will permit parts to release from the mold easily.

Typically, one to two degrees of drafts with an additional 1.5 degrees per 0.25mm depth of texture is enough to do the trick.

The mold part line will need to be located in a way that splits the draft in order to minimize it.

If no draft is ac­ceptable due to design considerations, a side action mold may be required.

STRATASYSDIRECT.COM | 888-311-10176At corners, the suggested inside radius is 0.5 times the material thickness and the outside radius is 1.5 times the material thickness.

A bigger radius should be used if part design allows.­­Figure 14: Radius RecommendationINSERTSInserts used in plastic parts provide a place for fasteners such as machine screws.

The advantage of using inserts is that they are often made of brass and are robust.

They allow for a great many cycles of assembly and disas-sembly.

Inserts are installed in injection molded parts us-ing one of the following methods:­­­­­­­­TEXTURES AND LETTERINGWhether to incorporate identifying information or to in-clude as an aesthetic addition, textures and lettering can be included onto mold surfaces for the end user or fac-tory purposes.

Texturing may also hide surface defects such as knit lines and other imperfections.

The depth of the texture or letters is somewhat limited, and extra draft needs to be provided to allow for part removal from the mold without dragging or marring the part.

Draft for texturing is somewhat dependent on the part design and specific texture desired.

As a general guide-line, 1.5° min.

per 0.025mm (0.001 inch) depth of tex-ture needs to be allowed for in addition to the normal draft.

Usually for general office equipment such as lap-top computers a texture depth of 0.025 mm (0.001 inch) is used and the minimum draft recommended is 1.5°.

More may be needed for heavier textured surfaces such as leather (with a depth of 0.125 mm/0.005 inch) that requires a minimum draft of 7.5°.

SHARP CORNERSSharp corners greatly increase stress concentration, which, when high enough, can lead to part failure.

Sharp corners often come about in non-obvious places, such as a boss attached to a surface, or a strengthening rib.

The radius of sharp corners needs to be watched closely because the stress concentra-tion factor varies with radius for a given thickness.

As illustrated in the chart to the left, the stress concen-tration factor is high for R/T values less than 0.5, but for R/T values over 0.5 the concentration lowers.

The stress concentration factor is a multiplier that greatly increases stress.

It is recommended that an inside radius be a minimum of one times the thickness.

In addition to reducing stresses, the fillet r adius p ro-vides a streamlined flow path for the molten plastic, re-sulting in an easier fill of the mold.

Figure 13: Stress Concentration Factor, K

7Figure 15: Threaded InsertLIVING HINGESLiving hinges are thin sections of plastic that connect two segments of a part to keep them together and allow the part to “hinge” open and closed.

Typically these hinges are incorporated in containers that are used in high vol­ume applications such as toolboxes and CD cases.

Figure 16: Box with Living HingeFigure 17: Living Hinge Design for Polypropylene and PolyethyleneMaterials used in molding living hinges must be very flexible, such as polypropylene or polyethylene.

A well-designed living hinge typically flexes more than a million cycles without failure.

ULTRASONIC INSERTIONUltrasonic insertion is when an insert is “vibrated” into place by using an ultrasonic transducer called the “horn” that is mounted into the ultrasonic device.

For optimum performance, the horn is specially designed for each ap­plication.

Ultrasonic energy is converted to thermal en­ergy by the vibrating action, which allows the insert to be melted into the hole.

This type of insertion can be done rapidly, with short cycle times, and low residual stresses.

Good melt flow characteristics for the plastic is neces­sary for the process to be successful.

THERMAL INSERTIONThis method uses a heated tool, like a soldering iron, to first heat the insert until it melts the plastic, and then presses the insert into place.

As the plastic cools it shrinks around the insert, capturing it.

The advantage of this method is that the special tooling is inexpensive and simple to use.

Care does need to be taken not to overheat the insert or plastic, which could result in a non-secure fit and degradation of the plastic.

MOLDED-INTo mold inserts into place during the molding cycle, core pins are used to hold the inserts.

The injected plastic completely encases the insert, which provides excellent retention.

This process may slow the molding cycle be­cause inserts have to be hand loaded, but it also elimi­nates secondary operations such as the ultrasonic and thermal insertion methods.

Finally, for high volume pro­duction runs, an automatic tool can load the inserts but this increases the complexity and cost of the mold.

STRATASYSDIRECT.COM | 888-311-10178­GAS ASSIST MOLDINGThis process is used to hollow out thick sections of a part where coring is not an option and sink is not accept-able.

Gas assist molding can be applied to almost any thermoplastic, and most conventional molding machines can be adapted for gas assist molding.­Figure 18: Gas Assist MoldingOVERMOLDINGThe overmolding process is when a flexible material is molded onto a more rigid material called a substrate.

If properly selected, the overmolded (flexible material) will form a strong bond with the substrate.

Bonding agents are no longer required to achieve optimum bond be-tween the substrate and overmold.

INSERT MOLDING The most widely used overmolding process is insert molding.

This is where a pre-molded substrate is placed into a mold and the flexible material is shot directly over it.

The advantage of this process is that conventional, single shot injection molding machines can be used.

TWO SHOT MOLDINGThis is a multi-material overmolding process that requires a special injection molding machine that incorporates two or more barrels.

This allows two or more materials to be shot into the same mold during the same molding cycle.

The two shot molding is usually associated with high volume production of greater than 250,000 cycles.

Copyright © 2015 Stratasys Direct, Inc.

All rights reserved.

Proprietary information do not distribute without prior consent from Stratasys Direct Manufacturing.

'''
    )
    abc.json_from_string('The bending radius of the part should be no less than four times the material thickness')