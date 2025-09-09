
def generate_slides_gamma(info: dict, selectedTemplate: str):
    """Generate slides with Gamma AI and return proper format for frontend"""
    
    # Validate required information
    presentation_title = info.get('presentation_title', 'Pitch Deck')
    business_name = info.get('business_name', 'Business')
    target_audience = info.get('target_audience', 'investors')
    industry = info.get('industry', 'technology')
    problem_statement = info.get('problem_statement', '')
    solution = info.get('solution', '')
    business_model = info.get('business_model', '')
    market_size = info.get('market_size', '')
    funding_ask = info.get('funding_ask', '')
    
    # Create a detailed prompt for Gamma AI
    input_text = f"""Create a professional pitch deck for "{business_name}".

Title: {presentation_title}
Industry: {industry}
Target audience: {target_audience}

Problem: {problem_statement}
Solution: {solution}
Business Model: {business_model}
Market Size: {market_size}
Funding Ask: {funding_ask}

Include standard pitch deck sections like problem, solution, market size, business model, competition, team, and funding ask.
Make it visually appealing and professional for {target_audience}."""

    # Set up the API call to Gamma
    gamma_api_url = "https://public-api.gamma.app/v0.2/generations"
    
    # Headers with API key
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": GAMMA_API_KEY
    }
    
    # Prepare the request body
    request_body = {
        "inputText": input_text,
        "textMode": "condense",
        "format": "presentation",
        "themeName": selectedTemplate,  # Professional theme
        "numCards": 10,
        "cardSplit": "auto",
        "exportAs": "pptx",
        "textOptions": {
            "amount": "brief",
            "language": "en"
        },
        "imageOptions": {
            "source": "aiGenerated",
            "model": "imagen-4-pro",
            "style": "photorealistic"
        },
        "cardOptions": {
            "dimensions": "fluid"
        },
        "sharingOptions": {
            "workspaceAccess": "fullAccess",
            "externalAccess": "edit"
        }
    }
    
    try:
        print(f"[DEBUG] Generating slides with Gamma AI...")
        print(f"[DEBUG] Business: {business_name}")
        print(f"[DEBUG] Title: {presentation_title}")
        print(f"[DEBUG] Industry: {industry}")
        print(f"[DEBUG] Audience: {target_audience}")
        print(f"[DEBUG] Prompt length: {len(input_text)}")
        
        # Step 1: Make the initial API call to Gamma
        response = requests.post(gamma_api_url, json=request_body, headers=headers, timeout=120)
        
        # Check for HTTP errors
        if response.status_code not in [200, 201]:  # Accept both 200 and 201 as success
            error_msg = f"Gamma API returned status code {response.status_code}: {response.text}"
            print(f"[DEBUG] {error_msg}")
            return {
                "type": "error",
                "message": f"I encountered an issue generating your slides: The presentation service returned an error. Please try again with different specifications.",
                "slides_url": None,
                "business_info": info
            }
            
        # Parse response
        response_data = response.json()
        print(f"[DEBUG] Initial response data: {response_data}")
        
        # Check for generation ID
        if "generationId" not in response_data:
            error_msg = "No generation ID returned from Gamma API"
            print(f"[DEBUG] Gamma API error: {error_msg}")
            return {
                "type": "error",
                "message": f"I encountered an issue generating your slides: The presentation service didn't return a valid generation ID. Please try again later.",
                "slides_url": None,
                "business_info": info
            }
        
        # Step 2: Get the generation ID
        generation_id = response_data["generationId"]
        
        # Step 3: Poll the status endpoint until generation is complete
        status_url = f"https://public-api.gamma.app/v0.2/generations/{generation_id}"
        
        print(f"[DEBUG] Generation started. ID: {generation_id}")
        print(f"[DEBUG] Polling status URL: {status_url}")
        
        # Set up polling parameters
        max_attempts = 12  # Maximum number of polling attempts
        poll_interval = 8  # Time between polling attempts in seconds
        attempts = 0
        
        gamma_url = f"https://gamma.app/view/{generation_id}"  # Default view URL
        export_url = None  # Will hold the downloadable PPTX URL when available
        
        # Loop until generation is complete or max attempts reached
        while attempts < max_attempts:
            attempts += 1
            print(f"[DEBUG] Polling attempt {attempts}/{max_attempts}...")
            
            # Make the status request
            status_response = requests.get(status_url, headers=headers, timeout=30)
            
            if status_response.status_code != 200:
                print(f"[DEBUG] Status check failed with code {status_response.status_code}: {status_response.text}")
                if attempts == max_attempts:
                    return {
                        "type": "error",
                        "message": f"I'm having trouble checking the status of your presentation. You can try viewing it directly at {gamma_url}",
                        "slides_url": gamma_url,
                        "business_info": info
                    }
                continue  # Try again
                
            # Parse status response
            status_data = status_response.json()
            print(f"[DEBUG] Status data: {status_data}")
            
            # Check if generation is complete
            status = status_data.get("status", "").lower()
            print(f"[DEBUG] Current status: {status}")
            
            if status == "completed":
                # Get the export URL if available
                export_url = status_data.get("exportUrl")
                if export_url:
                    print(f"[DEBUG] Export URL found: {export_url}")
                break
            
            elif status == "failed":
                error_message = status_data.get("errorMessage", "Unknown error")
                print(f"[DEBUG] Generation failed: {error_message}")
                return {
                    "type": "error",
                    "message": f"I encountered an issue generating your slides: {error_message}. Please try again with different specifications.",
                    "slides_url": None,
                    "business_info": info
                }
            
            elif status == "pending" or status == "running":
                print(f"[DEBUG] Generation still in progress, status: {status}")
                # Wait before polling again
                time.sleep(poll_interval)
                continue
                
            else:
                print(f"[DEBUG] Unknown status: {status}")
                if attempts == max_attempts:
                    break
                time.sleep(poll_interval)
        
        # Check if we've hit max attempts without completion
        if attempts >= max_attempts and status != "completed":
            print("[DEBUG] Maximum polling attempts reached without completion")
            return {
                "type": "partial_success",
                "message": f"Your presentation is still being generated. It may take a few more moments to complete to show here.",
                "slides_url": gamma_url,
                "business_info": info
            }
            
        # Success! Create success message
        success_message = f"🎉 **Your {presentation_title} pitch deck for {business_name} is ready!**\n\nI've created a professional pitch deck targeting {target_audience} in the {industry} industry. Click the link below to access your presentation."
        
        # We'll use the export URL if available, otherwise the view URL
        final_url = export_url or gamma_url
        
        print(f"[DEBUG] Successfully generated slides!")
        print(f"[DEBUG] Final URL: {final_url}")
        
        # Return successful result
        return {
            "type": "slides_generated",
            "message": success_message,
            "slides_url": final_url,
            "export_url": export_url,  # Include export URL separately in case frontend wants to offer download option
            "view_url": gamma_url,     # Include view URL separately for online viewing
            "business_info": {
                "business_name": business_name,
                "presentation_title": presentation_title,
                "target_audience": target_audience,
                "industry": industry,
                "problem_statement": problem_statement,
                "solution": solution,
                "business_model": business_model,
                "market_size": market_size,
                "funding_ask": funding_ask
            }
        }
    
    except Exception as e:
        print(f"[DEBUG] Gamma AI generation error: {str(e)}")
        
        # Return error result
        return {
            "type": "error",
            "message": f"I encountered an issue generating your slides: {str(e)}. Let me try again with different specifications, or you can adjust your requirements.",
            "slides_url": None,
            "business_info": info
        }





def generate_slides_gamma_dummy(info: dict, selectedTemplate: str):
    """Generate slides with Gamma AI and return proper format for frontend - TESTING VERSION"""
    
    # Extract basic info for logging
    presentation_title = info.get('presentation_title', 'Pitch Deck')
    business_name = info.get('business_name', 'Business')
    target_audience = info.get('target_audience', 'investors')
    industry = info.get('industry', 'technology')
    
    print(f"[DEBUG] TESTING MODE - Simulating slide generation...")
    print(f"[DEBUG] Business: {business_name}")
    print(f"[DEBUG] Title: {presentation_title}")
    print(f"[DEBUG] Industry: {industry}")
    print(f"[DEBUG] Audience: {target_audience}")
    
    # Simulate some processing time
    import time
    time.sleep(2)  # Brief delay to simulate generation
    
    # Return fixed test URL
    test_url = "https://gamma.app/view/AMpuazo32M2jKvCDs9i83"
    success_message = f"🎉 **Your {presentation_title} pitch deck for {business_name} is ready!**\n\nI've created a professional pitch deck targeting {target_audience} in the {industry} industry. Click the link below to access your presentation."
    
    print(f"[DEBUG] TESTING MODE - Returning fixed URL: {test_url}")
    
    return {
        "type": "slides_generated",
        "message": success_message,
        "slides_url": test_url,
        "export_url": test_url,
        "view_url": test_url,
        "business_info": {
            "business_name": business_name,
            "presentation_title": presentation_title,
            "target_audience": target_audience,
            "industry": industry,
            "problem_statement": info.get('problem_statement', ''),
            "solution": info.get('solution', ''),
            "business_model": info.get('business_model', ''),
            "market_size": info.get('market_size', ''),
            "funding_ask": info.get('funding_ask', '')
        }
    }