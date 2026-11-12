--[[
	The Weight of Forgotten Oaths - R6 cloak setup
	Run once from the Studio command bar (SurfaceAppearance maps can only be set by Studio/plugins).

	1. Import export/ForgottenOathsCloak.fbx with the 3D Importer (Scale Unit: Studs).
	   You get a Model with four MeshParts: CloakCape, CloakCowl, CloakDrapes, CloakHardware.
	2. Upload the eight PNGs from textures/roblox_1024 (Asset Manager > Bulk Import) and paste
	   their asset ids into TEXTURES below.
	3. Select the imported model (optionally also an R6 rig) and run this script.
	   It builds Accessory "ForgottenOathsCloak" in ServerStorage and, if a rig is selected, equips it.
]]

local TEXTURES = {
	FabricColor = "rbxassetid://0",       -- Cloak_Fabric_Color.png (alpha = torn cut-outs)
	FabricNormal = "rbxassetid://0",      -- Cloak_Fabric_Normal.png
	FabricRoughness = "rbxassetid://0",   -- Cloak_Fabric_Roughness.png
	FabricEmissive = "rbxassetid://0",    -- Cloak_Fabric_Emissive.png (rune glow mask)
	HardwareColor = "rbxassetid://0",     -- Cloak_Hardware_Color.png
	HardwareNormal = "rbxassetid://0",    -- Cloak_Hardware_Normal.png
	HardwareRoughness = "rbxassetid://0", -- Cloak_Hardware_Roughness.png
	HardwareMetalness = "rbxassetid://0", -- Cloak_Hardware_Metalness.png
}

-- Generated from export/roblox_parts.json: mesh bounding-box size and centre offset from the
-- R6 Torso centre, in studs (Torso space: +X right, +Y up, -Z forward).
local PARTS = {
	CloakCape = { size = Vector3.new(5.5326, 3.7303, 1.7839), offset = Vector3.new(0.0111, -0.7832, 1.0046) }, -- 13320 tris
	CloakCowl = { size = Vector3.new(2.4897, 1.5395, 2.1623), offset = Vector3.new(0.0015, 0.8709, 0.2701) }, -- 11004 tris
	CloakDrapes = { size = Vector3.new(3.1373, 3.1395, 2.3352), offset = Vector3.new(0.1486, -0.5118, 0.4688) }, -- 7748 tris
	CloakHardware = { size = Vector3.new(2.1710, 2.5149, 2.2720), offset = Vector3.new(-0.0262, 0.1976, 0.2343) }, -- 14236 tris
}

local RUNE_TINT = Color3.fromRGB(199, 178, 255)
local RUNE_STRENGTH = 2

local Selection = game:GetService("Selection")
local ServerStorage = game:GetService("ServerStorage")

local source, rig
for _, inst in ipairs(Selection:Get()) do
	if inst:FindFirstChildOfClass("Humanoid") then
		rig = inst
	elseif inst:FindFirstChild("CloakCape", true) then
		source = inst
	end
end
assert(source, "Select the imported model that contains CloakCape, CloakCowl, CloakDrapes and CloakHardware")

local function applyAppearance(part, isHardware)
	for _, child in ipairs(part:GetChildren()) do
		if child:IsA("SurfaceAppearance") then
			child:Destroy()
		end
	end
	local sa = Instance.new("SurfaceAppearance")
	if isHardware then
		sa.ColorMap = TEXTURES.HardwareColor
		sa.NormalMap = TEXTURES.HardwareNormal
		sa.RoughnessMap = TEXTURES.HardwareRoughness
		sa.MetalnessMap = TEXTURES.HardwareMetalness
	else
		sa.ColorMap = TEXTURES.FabricColor
		sa.NormalMap = TEXTURES.FabricNormal
		sa.RoughnessMap = TEXTURES.FabricRoughness
		-- alpha in the colour map cuts out the tears and holes; Transparency 0 keeps them crisp
		sa.AlphaMode = Enum.AlphaMode.Transparency
		local ok = pcall(function()
			sa.EmissiveMaskContent = Content.fromUri(TEXTURES.FabricEmissive)
			sa.EmissiveTint = RUNE_TINT
			sa.EmissiveStrength = RUNE_STRENGTH
		end)
		if not ok then
			warn("Emissive masks are not available in this Studio build; the rune is still painted in the colour map.")
		end
	end
	sa.Parent = part
end

local accessory = Instance.new("Accessory")
accessory.Name = "ForgottenOathsCloak"
accessory.AccessoryType = Enum.AccessoryType.Back

local handleInfo = PARTS.CloakCape
local handle
for name, info in pairs(PARTS) do
	local src = source:FindFirstChild(name, true)
	assert(src and src:IsA("MeshPart"), "Missing MeshPart " .. name)
	local part = src:Clone()
	part.Size = info.size -- also corrects any import scale mismatch
	part.Anchored = false
	part.CanCollide = false
	part.CanTouch = false
	part.CanQuery = false
	part.Massless = true
	part.CastShadow = true
	part.Transparency = 0
	applyAppearance(part, name == "CloakHardware")
	if name == "CloakCape" then
		part.Name = "Handle"
		handle = part
	end
	part.Parent = accessory
	PARTS[name].instance = part
end

handle.CFrame = CFrame.new()
for name, info in pairs(PARTS) do
	if info.instance ~= handle then
		info.instance.CFrame = handle.CFrame * CFrame.new(info.offset - handleInfo.offset)
		local weld = Instance.new("WeldConstraint")
		weld.Part0 = handle
		weld.Part1 = info.instance
		weld.Parent = info.instance
	end
end

-- R6 Torso has BodyBackAttachment at (0, 0, 0.5); line the cloak up against it.
local attachment = Instance.new("Attachment")
attachment.Name = "BodyBackAttachment"
attachment.CFrame = CFrame.new(Vector3.new(0, 0, 0.5) - handleInfo.offset)
attachment.Parent = handle

accessory.Parent = ServerStorage
print("Created ServerStorage." .. accessory.Name)

if rig then
	local humanoid = rig:FindFirstChildOfClass("Humanoid")
	local torso = rig:FindFirstChild("Torso")
	local copy = accessory:Clone()
	local ok = pcall(function()
		humanoid:AddAccessory(copy)
	end)
	if not ok and torso then
		-- fallback: weld straight to the Torso
		copy.Parent = rig
		local h = copy:FindFirstChild("Handle")
		h.CFrame = torso.CFrame * CFrame.new(handleInfo.offset)
		local weld = Instance.new("Weld")
		weld.Part0 = torso
		weld.Part1 = h
		weld.C0 = CFrame.new(handleInfo.offset)
		weld.Parent = h
	end
	print("Equipped on " .. rig.Name)
end
