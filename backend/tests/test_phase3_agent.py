from querying.agents.factory import create_phase3_agent

# Test Phase III agent with real data
print("Testing Phase III agent with real data...")

try:
    agent = create_phase3_agent()
    print(f"✅ Phase III agent created with {agent.registry.get_plugin_count()} plugins")
    
    # Test 1: Basic metadata query (should work with existing data)
    print("\n🔍 Test 1: Basic metadata query")
    result = agent.process_query("find all PDF files")
    print(f"Result: {result[:200]}..." if len(result) > 200 else result)
    
    # Test 2: Enhanced metadata query
    print("\n🔍 Test 2: Enhanced metadata query")
    result = agent.process_query("show me the latest 3 files")
    print(f"Result: {result[:200]}..." if len(result) > 200 else result)
    
    # Test 3: Document similarity (if supported)
    print("\n🔍 Test 3: Document relationships")
    result = agent.process_query("find documents similar to budget")
    print(f"Result: {result[:200]}..." if len(result) > 200 else result)
    
    # Test 4: Simple reporting
    print("\n🔍 Test 4: Collection summary")
    result = agent.process_query("generate a summary of the document collection")
    print(f"Result: {result[:300]}..." if len(result) > 300 else result)
    
    print("\n✅ Phase III agent testing complete!")
    
except Exception as e:
    print(f"❌ Error testing Phase III agent: {e}")
    import traceback
    traceback.print_exc()
