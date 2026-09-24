const fs=require('fs');
const {Document,Packer,Paragraph,TextRun,AlignmentType,LevelFormat,BorderStyle,TabStopType}=require('docx');
const NAVY="1F3A5F", GREY="555555", FONT="Calibri";
const W=12240, M=1008;
const t=(s,o={})=>new TextRun({text:s,font:FONT,size:o.size||20,bold:o.bold,italics:o.italics,color:o.color});
const H=(s)=>new Paragraph({spacing:{before:200,after:80},border:{bottom:{style:BorderStyle.SINGLE,size:6,color:NAVY,space:2}},children:[t(s,{bold:true,size:22,color:NAVY})]});
const B=(s,lead)=>new Paragraph({numbering:{reference:"b",level:0},spacing:{after:50},children: lead?[t(lead,{bold:true}),t(s)]:[t(s)]});
const P=(runs,o={})=>new Paragraph({spacing:{after:o.after??80},alignment:o.align,children:runs});
const job=(title,org,loc,dates)=>[
  new Paragraph({spacing:{before:120,after:0},tabStops:[{type:TabStopType.RIGHT,position:W-2*M}],children:[t(title,{bold:true,size:21}),t("  |  "+org,{bold:true,size:21,color:NAVY}),t("\t"+dates,{size:19,color:GREY})]}),
  P([t(loc,{italics:true,size:19,color:GREY})],{after:40})];

const kids=[
 P([t("LUCIAN BLAGA",{bold:true,size:40,color:NAVY})],{align:AlignmentType.CENTER,after:20}),
 P([t("DIRECTOR, VISUAL PRODUCTION  |  VIDEO · PHOTOGRAPHY · POST-PRODUCTION",{bold:true,size:21,color:GREY})],{align:AlignmentType.CENTER,after:40}),
 P([t("Las Vegas, NV   •   (936) 828-0110   •   lucian@arcatv.us   •   Portfolio: arca.tv/lucian   •   vionics.live/mgm",{size:19})],{align:AlignmentType.CENTER,after:120}),

 H("PROFILE"),
 P([t("Visual production leader with 25+ years directing video, photography and post-production for brands, gaming, hospitality and entertainment across the U.S., Europe and Australia. Hands-on on set and in the edit bay — camera, lighting, color, retouching and finishing — and a builder of teams. At Las Vegas Sands, built production studios in Las Vegas and Sofia, recruited and trained video professionals, and set the visual standard for live casino content that ran three years without an on-air interruption. Known for defining a clear look and feel, coaching shooters and editors to raise the bar, and delivering high volume without losing craft.")]),

 H("ALIGNMENT WITH THE ROLE"),
 B(" Define the look, lighting, color and finishing standard for all video and photo output across multiple brands, formats and channels.","Senior visual authority —"),
 B(" Recruited, hired and trained video teams on two continents; coach shooting technique, storytelling, editing and color.","Team quality & growth —"),
 B(" Directed commercials, music videos, documentaries and 4–12-camera live productions; the calm decision-maker on high-pressure shoots.","On-set execution —"),
 B(" Editing, color grading (DaVinci Resolve), retouching and final review; built repeatable pipelines from shoot to delivery.","Post-production quality —"),
 B(" Partner with producers, creative leadership, marketing and operations to turn briefs into finished, on-brand visual work.","Creative partnership —"),
 B(" 30–40% lower cost per deliverable, $4.5M+ cumulative savings, AI-assisted workflows that speed delivery while protecting quality.","Scale & efficiency —"),

 H("PROFESSIONAL EXPERIENCE"),
 ...job("Executive Director of Broadcast","Awevoke","Las Vegas, NV","Mar 2026 – Present"),
 B("Lead visual and broadcast production for live game-show studios — camera coverage, lighting, switching, graphics and on-air quality — aligning creative and operational teams on a single visual standard."),
 ...job("Founder & Chief Creative Technology Executive","Vionics","Global / Remote","Dec 2025 – Present"),
 B("Lead creative and technical direction for live production, virtual sets and real-time graphics; build AI-assisted and IP-video workflows that help teams create and deliver branded visual content at scale."),
 ...job("Global Director, Broadcast Production","Las Vegas Sands","Las Vegas, NV / Sofia, Bulgaria","Sep 2022 – Dec 2025"),
 B("Set the visual standard for all live casino content — lighting design, camera coverage, sets, motion graphics, color grading and final on-air quality across multi-camera productions."),
 B("Recruited, hired and trained video and broadcast professionals, including a 20-person audio/video team in Sofia; led a 35-person gaming media studio in Las Vegas, mentoring operators, editors and technical leads."),
 B("Built and launched production studios in Las Vegas and Sofia, directing set creation, camera and lighting systems, installation and production readiness, then replicating the operating model across sites."),
 B("Co-developed a purpose-built video-over-IP broadcast camera with Z CAM; engineered redundant, sub-second-latency workflows that ran three years without an on-air interruption."),
 B("Managed equipment, vendors and partners through design, procurement, installation and launch; led $40M+ in studio operations, cut cost per deliverable 30–40% and delivered $4.5M+ in cumulative savings."),
 ...job("Content Director / Creative Lead","Pink Cilantro","Houston, TX","Jan – Sep 2022"),
 B("Led branded content, commercials, filmed interviews, 3D animation and campaign assets from concept and pitch through shoot, edit and delivery."),
 B("Partnered with the CEO, agencies, producers, editors and crews to solve creative and on-set problems while hitting fast campaign deadlines."),
 ...job("VP, Creative & Video Production / Founder","Arca Productions & Arca TV","Houston, TX / Global","Jan 2014 – Jan 2022"),
 B("Founded and scaled a creative studio delivering commercials, brand storytelling, original programs, photography and live content across video, motion and digital channels."),
 B("Led ~30 videographers, editors, designers and collaborators across the U.S., Europe and Australia, setting visual standards and production workflows for distributed teams."),
 B("Built repeatable production and post pipelines — shoot, edit, color, finishing and delivery — for broadcast and digital; designed virtual studios and real-time 3D environments."),
 ...job("Founder & CEO, Creative & Media Operations","Magma Entertainment","Bucharest, Romania","2003 – 2014"),
 B("Built and operated two television stations and production studios, overseeing creative direction, staff, suppliers and budgets of ~$5M."),
 B("Directed award-winning music videos, commercials, documentaries, concerts and live TV (4–12 cameras); led creative, VFX and post-production teams; created a one-second TV commercial format."),
 ...job("VP Video Production / Partner","IZEXIM Group Media","Bucharest, Romania","1999 – 2003"),
 B("Directed First Division soccer broadcasts, TV programs, concerts and music videos, supervising 60+ creative and technical professionals."),
 B("Led visual direction on McCann Erickson campaigns for Gillette, Stella Artois and Coca-Cola."),

 H("CRAFT & TOOLS"),
 B(" Camera and lighting direction, multi-camera and live switching, on-set direction, talent coaching.","Video:"),
 B(" Brand, lifestyle, portrait and editorial direction; image selection and retouching (Adobe Photoshop, Illustrator).","Photography:"),
 B(" Premiere Pro, After Effects, Audition, DaVinci Resolve color grading; finishing and QC review.","Post-production:"),
 B(" Unreal Engine, Unity, Maya, 3ds Max, Nuke; generative image/video tools and AI-assisted workflows; NDI, Dante, video-over-IP.","3D, AI & systems:"),

 H("EDUCATION & RECOGNITION"),
 P([t("Bachelor’s Degree, Film & Media Studies   •   MTV Award – Best Music Video   •   Two APTR National Television Awards (Romania)")]),
];
const doc=new Document({numbering:{config:[{reference:"b",levels:[{level:0,format:LevelFormat.BULLET,text:"•",alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:300,hanging:220}}}}]}]},
 sections:[{properties:{page:{size:{width:W,height:15840},margin:{top:864,bottom:864,left:M,right:M}}},children:kids}]});
Packer.toBuffer(doc).then(b=>fs.writeFileSync("Lucian_Blaga_MGM_Director_Visual_Production.docx",b));
